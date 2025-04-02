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

let pulseRadius = POINT_RADIUS;
let pulseGrowing = true;
let pulseAnimationId = null;


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

function resetCalibration() {
    currentIndex = 0;
    capturedPoints = [];
    canvas.style.display = "none";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("processingMessage").style.display = "none";
    document.getElementById("startButton").style.display = "block";
}

async function startCalibration() {
    try {
        await enterFullScreen();
        updateCanvasSize();
        document.getElementById("startButton").style.display = "none";

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
    const [x, y] = calibrationPoints[index];
    let start = null;
    const duration = 1000;
    const maxPulse = POINT_RADIUS + 8;
    const minPulse = POINT_RADIUS - 2;

    function animate(timestamp) {
        if (!start) start = timestamp;
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);

        // Pulse effect: ease-out style shrink back
        const radius = maxPulse - (maxPulse - minPulse) * progress;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = "black";
        ctx.fill();

        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            // Draw final stable point
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = "black";
            ctx.fill();
        }
    }

    requestAnimationFrame(animate);
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

        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.display = "none";

        // Show wait message
        document.getElementById("processingMessage").style.display = "block";

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
                resetCalibration();
            } else {
                console.log("Calibration and fine-tuning completed:", result);
                document.getElementById("processingMessage").style.display = "none";
                startExperiments();
            }
        } catch (err) {
            console.error("Error sending calibration data:", err);
            showErrorMessage("Network error during calibration.");
            resetCalibration();
        }
    }
});

async function startExperiments() {
    canvas.style.display = "block";
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw central fixation cross
    drawFixationCross();

    let fixationFrames = 0;
    const requiredFixation = 15;

    async function checkFixationLoop() {
        if (!latestFaceLandmarks || latestFaceLandmarks.length !== 468) {
            requestAnimationFrame(checkFixationLoop);
            return;
        }

        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        const captureCtx = captureCanvas.getContext("2d");
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const imageBase64 = captureCanvas.toDataURL("image/jpeg");

        try {
            const formData = new FormData();
            formData.append("image_base64", imageBase64);
            formData.append("landmarks", JSON.stringify(latestFaceLandmarks));

            const response = await fetch("/model/predict_gaze", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (result.x_cm !== undefined && result.y_cm !== undefined) {
                const isLookingCenter = Math.abs(result.x_cm) < 2.0 && Math.abs(result.y_cm) < 2.0;

                if (isLookingCenter) {
                    fixationFrames++;
                    if (fixationFrames >= requiredFixation) {
                        console.log("User fixated the cross.");
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        // Display left and right images
                        const imageLeft = new Image();
                        const imageRight = new Image();

                        imageLeft.src = "/static/img/left.png";    
                        imageRight.src = "/static/img/right.png";

                        imageLeft.onload = () => {
                            imageRight.onload = () => {
                                const midY = canvas.height / 2;
                                const imageHeight = canvas.height * 0.4;
                                const imageWidth = canvas.width * 0.25;

                                const leftX = canvas.width * 0.15;
                                const rightX = canvas.width * 0.6;

                                ctx.clearRect(0, 0, canvas.width, canvas.height);
                                ctx.drawImage(imageLeft, leftX, midY - imageHeight / 2, imageWidth, imageHeight);
                                ctx.drawImage(imageRight, rightX, midY - imageHeight / 2, imageWidth, imageHeight);

                                startRealTimeGazePrediction();
                            };
                        };
                        return;
                    }
                } else {
                    fixationFrames = 0;
                }
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }

        requestAnimationFrame(checkFixationLoop);
    }

    checkFixationLoop();
}

function drawFixationCross() {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const crossSize = 40;
    const lineWidth = 6;

    ctx.strokeStyle = "black";
    ctx.lineWidth = lineWidth;

    ctx.beginPath();
    ctx.moveTo(centerX - crossSize, centerY);
    ctx.lineTo(centerX + crossSize, centerY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(centerX, centerY - crossSize);
    ctx.lineTo(centerX, centerY + crossSize);
    ctx.stroke();
}

function startRealTimeGazePrediction() {
    async function loop() {
        if (!latestFaceLandmarks || latestFaceLandmarks.length !== 468) {
            requestAnimationFrame(loop);
            return;
        }

        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        const captureCtx = captureCanvas.getContext("2d");
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const imageBase64 = captureCanvas.toDataURL("image/jpeg");

        try {
            const formData = new FormData();
            formData.append("image_base64", imageBase64);
            formData.append("landmarks", JSON.stringify(latestFaceLandmarks));

            const response = await fetch("/model/predict_gaze", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (result.x_cm !== undefined) {
                if (result.x_cm < 0) {
                    console.log("Looking LEFT");
                } else {
                    console.log("Looking RIGHT");
                }
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }

        requestAnimationFrame(loop);
    }

    loop();
}
