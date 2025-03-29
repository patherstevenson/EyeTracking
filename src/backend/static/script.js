const faceMesh = new FaceMesh({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

function showErrorMessage(message) {
    let errorDiv = document.getElementById("errorMessage");

    if (!errorDiv) {
        errorDiv = document.createElement("div");
        errorDiv.id = "errorMessage";
        document.body.appendChild(errorDiv);
    }

    errorDiv.textContent = message;
    errorDiv.style.position = "absolute";
    errorDiv.style.top = "2%";
    errorDiv.style.left = "50%";
    errorDiv.style.transform = "translateX(-50%)";
    errorDiv.style.padding = "15px 30px";
    errorDiv.style.backgroundColor = "black"
    errorDiv.style.color = "white";
    errorDiv.style.fontSize = "20px";
    errorDiv.style.fontWeight = "bold";
    errorDiv.style.textAlign = "center";
    errorDiv.style.borderRadius = "10px";
    errorDiv.style.zIndex = "1000";
    errorDiv.style.display = "block";
    errorDiv.style.maxWidth = "40%";

    setTimeout(() => {
        errorDiv.style.display = "none";
    }, 2500);
}


let SCREEN_WIDTH = window.screen.width;
let SCREEN_HEIGHT = window.screen.height

const canvas = document.getElementById("calibrationCanvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const RADIUS_RATIO = 0.006;
const POINT_RADIUS = SCREEN_WIDTH * RADIUS_RATIO;

let isFullScreen = false;
let needsFullScreenRestore = false;
let currentIndex = 0;
let capturedPoints = [];
let calibrationPoints = [];
let latestFaceLandmarks = null;
let lastLandmarkTimestamp = 0;

// Capturer la vidéo en arrière-plan SANS l'afficher
const video = document.createElement("video");
video.setAttribute("autoplay", "");
video.setAttribute("playsinline", "");
video.style.display = "none";
document.body.appendChild(video);

// Activer la webcam sans affichage
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => console.error("Error accessing webcam:", err));

// Analyser les images de la webcam en continu avec MediaPipe
async function processWebcamFrame() {
    if (!video.paused && !video.ended) {
        await faceMesh.send({ image: video });
    }
    requestAnimationFrame(processWebcamFrame);
}

// Lancer la détection MediaPipe en arrière-plan
faceMesh.onResults(results => {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length === 1) {
        latestFaceLandmarks = results.multiFaceLandmarks[0];
        lastLandmarkTimestamp = Date.now();
    } else {
        latestFaceLandmarks = null;
    }
});

requestAnimationFrame(processWebcamFrame);

// Détection sortie plein écran
document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) {
        console.warn("User exited fullscreen mode. Next click will restore fullscreen.");
        needsFullScreenRestore = true;
    }
});

// Bloquer la touche Échap pour éviter la sortie plein écran
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        needsFullScreenRestore = true;
        console.warn("Escape pressed: Next click will restore fullscreen.");
    }
});

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("startButton").addEventListener("click", startCalibration);
});

function updateCanvasSize() {
    canvas.width = SCREEN_WIDTH;
    canvas.height = SCREEN_HEIGHT;

    calibrationPoints = [
        [SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.1], 
        [SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.25],
        [SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.1],
        [SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.25],
        [SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.1],
        [SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.5],
        [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2],
        [SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.5],
        [SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.9],
        [SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.75],
        [SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.9],
        [SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.75],
        [SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.9]
    ];
}


async function startCalibration() {
    try {
        await enterFullScreen();
        updateCanvasSize();
        document.getElementById("startButton").style.display = "none";

        // Envoyer les dimensions au backend
        const formData = new FormData();
        formData.append("width", window.innerWidth);
        formData.append("height", window.innerHeight);

        const response = await fetch("/config/update_screen", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server failed to update screen size");
        }

        const result = await response.json();
        console.log("Screen size registered:", result);

        // Seulement après succès
        canvas.style.display = "block";
        drawCalibrationPoint(0);

    } catch (error) {
        console.error("Calibration start failed:", error);
        showErrorMessage("Failed to start calibration. Retrying...");

        // Re-essaye après 1 seconde
        setTimeout(startCalibration, 1000);
    }
}


function enterFullScreen() {
    const docElm = document.documentElement;
    if (docElm.requestFullscreen) {
        return docElm.requestFullscreen();
    } else if (docElm.mozRequestFullScreen) {
        return docElm.mozRequestFullScreen();
    } else if (docElm.webkitRequestFullscreen) {
        return docElm.webkitRequestFullscreen();
    } else if (docElm.msRequestFullscreen) {
        return docElm.msRequestFullscreen();
    }
    return Promise.reject("Fullscreen not supported");
}

function drawCalibrationPoint(index) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    const [x, y] = calibrationPoints[index];
    ctx.beginPath();
    ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
}

canvas.addEventListener("click", async function(event) {
    if (needsFullScreenRestore) {
        enterFullScreen();
        needsFullScreenRestore = false;
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const [targetX, targetY] = calibrationPoints[currentIndex];
    const distance = Math.sqrt((x - targetX) ** 2 + (y - targetY) ** 2);

    if (distance > POINT_RADIUS) {
        showErrorMessage("Click inside the black circle");
        return;
    }

    const now = Date.now();
    const isRecent = (now - lastLandmarkTimestamp) < 500;

    if (!latestFaceLandmarks || !isRecent) {
        showErrorMessage("Face not detected properly. Ensure only one face is visible!");
        return;
    }

    console.log(`Point ${currentIndex} clicked at (${x}, ${y})`);

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext("2d");
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const imageBase64 = captureCanvas.toDataURL("image/jpeg");

    capturedPoints.push({
        x_pixel: Math.floor(x),
        y_pixel: Math.floor(y),
        image_base64: imageBase64,
        landmarks: latestFaceLandmarks
    });

    currentIndex++;
    if (currentIndex < calibrationPoints.length) {
        drawCalibrationPoint(currentIndex);
    } else {
        console.log("Collected 13 points. Sending calibration data...");

        try {
            const response = await fetch("/calibration/submit_calibration", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ points: capturedPoints })
            });

            const result = await response.json();
            if (result.error) {
                console.error("Calibration error:", result.error);
                showErrorMessage("Calibration failed.");
            } else {
                console.log("Calibration and fine-tuning completed:", result);
            }
        } catch (err) {
            console.error("Error sending calibration data:", err);
            showErrorMessage("Network error during calibration.");
        }
    }
});
