from flask import Flask, render_template, request, redirect
import whisper

app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            audio = whisper.load_audio(file)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")

            # decode the audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)
            print(f"Detected language: {max(probs, key=probs.get)}")

            # decode the audio
            options = whisper.DecodingOptions()
            transcript = whisper.decode(model, mel, options)

    return render_template('index.html', transcript=transcript)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
