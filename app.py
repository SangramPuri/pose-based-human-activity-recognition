from pathlib import Path
import uuid
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
app=Flask(__name__); UPLOAD_DIR=Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True); ALLOWED={".mp4",".avi",".mov"}
@app.route("/")
def index(): return render_template("index.html")
@app.route("/upload",methods=["POST"])
def upload():
    video=request.files.get("video")
    if not video or not video.filename: return render_template("index.html",error="Please select a video.")
    if Path(video.filename).suffix.lower() not in ALLOWED: return render_template("index.html",error="Supported formats: MP4, AVI and MOV.")
    name=f"{uuid.uuid4().hex[:10]}_{secure_filename(video.filename)}"; video.save(UPLOAD_DIR/name)
    return render_template("index.html",message="Video uploaded. Connect the trained pose backend and model for inference.",filename=name)
@app.route("/uploads/<path:filename>")
def uploaded_file(filename): return send_from_directory(UPLOAD_DIR,filename)
if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=True)
