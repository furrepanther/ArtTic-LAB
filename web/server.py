import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from core import logic as core
from core.logic import OOMError
import os

APP_LOGGER_NAME = "arttic_lab"
logger = logging.getLogger(APP_LOGGER_NAME)

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/fonts", StaticFiles(directory="web/fonts"), name="fonts")
app.mount("/icons", StaticFiles(directory="web/icons"), name="icons")

env = Environment(loader=FileSystemLoader("web/templates"))
index_template = env.get_template("index.html")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return index_template.render()


@app.get("/api/status")
async def get_status():
    return core.get_app_status()


@app.get("/api/config")
async def get_initial_config():
    return core.get_config()


@app.get("/api/gallery")
async def get_gallery_images():
    return {"images": core.get_output_images()}


@app.get("/api/prompts")
async def get_prompts():
    return core.get_prompts()


@app.get("/api/image_metadata/{filename}")
async def get_image_metadata(filename: str):
    return core.get_image_metadata(filename)


@app.post("/api/prompts")
async def add_prompt(request: Request):
    body = await request.json()
    return core.add_prompt(
        body.get("title"), body.get("prompt"), body.get("negative_prompt", "")
    )


@app.put("/api/prompts")
async def update_prompt(request: Request):
    body = await request.json()
    return core.update_prompt(
        body.get("old_title"),
        body.get("new_title"),
        body.get("prompt"),
        body.get("negative_prompt"),
    )


@app.delete("/api/prompts")
async def delete_prompt(request: Request):
    body = await request.json()
    return core.delete_prompt(body.get("title"))


class ConnectionManager:
    def __init__(self):
        self.connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: dict):
        for c in self.connections[:]:
            try:
                await c.send_json(msg)
            except RuntimeError:
                self.disconnect(c)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        loop = asyncio.get_running_loop()
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            payload = data.get("payload", {})

            async def progress_cb(p, d):
                try:
                    await websocket.send_json(
                        {
                            "type": "progress_update",
                            "data": {"progress": p, "description": d},
                        }
                    )
                except RuntimeError:
                    pass

            try:
                if action == "load_model":
                    res = await asyncio.to_thread(
                        core.load_model,
                        **payload,
                        progress_callback=progress_cb,
                        loop=loop,
                    )
                    await websocket.send_json({"type": "model_loaded", "data": res})

                elif action == "generate_image":
                    try:
                        res = await asyncio.to_thread(
                            core.generate_image,
                            **payload,
                            progress_callback=progress_cb,
                            loop=loop,
                        )
                        await websocket.send_json(
                            {"type": "generation_complete", "data": res}
                        )
                        await manager.broadcast(
                            {
                                "type": "gallery_updated",
                                "data": {"images": core.get_output_images()},
                            }
                        )
                    except OOMError as e:
                        await websocket.send_json(
                            {"type": "generation_failed", "data": {"message": str(e)}}
                        )

                elif action == "unload_model":
                    res = await asyncio.to_thread(core.unload_model)
                    await websocket.send_json({"type": "model_unloaded", "data": res})

                elif action == "toggle_share":
                    res = await asyncio.to_thread(core.toggle_share)
                    await websocket.send_json({"type": "share_toggled", "data": res})

                elif action == "get_settings_data":
                    data = {
                        "models": await asyncio.to_thread(core.get_model_files),
                        "loras": await asyncio.to_thread(core.get_lora_files),
                    }
                    await websocket.send_json({"type": "settings_data", "data": data})

                elif action == "restart_backend":
                    await websocket.send_json(
                        {"type": "backend_restarting", "data": {}}
                    )
                    await asyncio.sleep(0.5)
                    core.restart_backend()
                    break

                elif action == "clear_cache":
                    res = await asyncio.to_thread(core.clear_cache)
                    await websocket.send_json({"type": "cache_cleared", "data": res})

                elif action == "delete_image":
                    res = await asyncio.to_thread(
                        core.delete_image, payload.get("filename")
                    )
                    await websocket.send_json({"type": "image_deleted", "data": res})
                    await manager.broadcast(
                        {
                            "type": "gallery_updated",
                            "data": {"images": core.get_output_images()},
                        }
                    )

                elif action in ["delete_model_file", "delete_lora_file"]:
                    func = (
                        core.delete_model_file
                        if action == "delete_model_file"
                        else core.delete_lora_file
                    )
                    res = await asyncio.to_thread(func, payload.get("filename"))
                    await websocket.send_json(
                        {
                            "type": f"{'model' if 'model' in action else 'lora'}_file_deleted",
                            "data": res,
                        }
                    )
                    if res.get("status") == "success":
                        data = {
                            "models": await asyncio.to_thread(core.get_model_files),
                            "loras": await asyncio.to_thread(core.get_lora_files),
                        }
                        await manager.broadcast(
                            {"type": "settings_data_updated", "data": data}
                        )

            except Exception as e:
                # Catch RuntimeError explicitly to handle client disconnects during task processing
                if 'Cannot call "send"' in str(e):
                    logger.info(f"Client disconnected during {action}.")
                    break

                logger.error(f"Error in {action}: {e}", exc_info=True)
                try:
                    await websocket.send_json(
                        {"type": "error", "data": {"message": str(e)}}
                    )
                except RuntimeError:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        manager.disconnect(websocket)
