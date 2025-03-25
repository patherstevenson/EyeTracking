let SCREEN_WIDTH = window.innerWidth;
let SCREEN_HEIGHT = window.innerHeight;
const canvas = document.getElementById("calibrationCanvas");
const ctx = canvas.getContext("2d");
canvas.width = SCREEN_WIDTH;
canvas.height = SCREEN_HEIGHT;

let isFullScreen = false;
let needsFullScreenRestore = false;

let currentIndex = 0;
let capturedPoints = [];
let calibrationPoints = [];

document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) {  
        console.warn("User exited fullscreen mode. Next click will restore fullscreen.");
        needsFullScreenRestore = true; 
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        needsFullScreenRestore = true;
        console.warn("Escape pressed: Next click will restore fullscreen.");
    }
});

function updateScreenSize() {
    SCREEN_WIDTH = window.innerWidth;
    SCREEN_HEIGHT = window.innerHeight;
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

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("startButton").addEventListener("click", startCalibration);
});

function startCalibration() {
    enterFullScreen().then(() => {
        updateScreenSize(); 
        document.getElementById("startButton").style.display = "none";
        canvas.style.display = "block";

        drawCalibrationPoint(0);
    }).catch(err => console.warn("Fullscreen error:", err));
}

function enterFullScreen() {
    const docElm = document.documentElement;
    if (docElm.requestFullscreen) {
        return docElm.requestFullscreen();
    } else if (docElm.mozRequestFullScreen) { // Firefox
        return docElm.mozRequestFullScreen();
    } else if (docElm.webkitRequestFullscreen) { // Chrome, Safari
        return docElm.webkitRequestFullscreen();
    } else if (docElm.msRequestFullscreen) { // IE/Edge
        return docElm.msRequestFullscreen();
    }
    return Promise.reject("Fullscreen not supported");
}

function drawCalibrationPoint(index) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "red";
    const [x, y] = calibrationPoints[index];
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();
}

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
    errorDiv.style.backgroundColor = "rgba(255, 0, 0, 0.9)";
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

canvas.addEventListener("click", function(event) {
    if (needsFullScreenRestore) {  
        console.warn("Restoring fullscreen mode...");
        enterFullScreen();
        needsFullScreenRestore = false; 
        return;  
    }

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const [targetX, targetY] = calibrationPoints[currentIndex];

    const distance = Math.sqrt((x - targetX) ** 2 + (y - targetY) ** 2);

    if (distance <= 20) { 
        console.log(`Point ${currentIndex} correctly clicked at (${x}, ${y})`);
        capturedPoints.push({ x, y });
        currentIndex++;

        if (currentIndex < calibrationPoints.length) {
            drawCalibrationPoint(currentIndex);
        } else {
            sendCalibrationData();
        }
    } else {
        console.log(`Incorrect click at (${x}, ${y}) - Expected near (${targetX}, ${targetY})`);
        showErrorMessage("Click inside the red circle!");
    }
});

async function sendCalibrationData() {
    console.log("Sending calibration data...");
    const response = await fetch("/calibration/submit_calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(capturedPoints)
    });

    const result = await response.json();
    console.log("Server response:", result);
    alert("Calibration completed successfully!");
}
