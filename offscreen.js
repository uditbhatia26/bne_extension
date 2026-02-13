// Offscreen document: handles actual tab capture and recording

let mediaRecorder = null;
let recordedChunks = [];
let currentMode = "tab";
let capturedFrames = [];
let currentStream = null;
let recordingStartTime = null;

console.log("[Offscreen] Script loaded and ready");

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("[Offscreen] Received message:", message.type, "target:", message.target);

    // Only handle messages targeted at offscreen
    if (message.target !== "offscreen") {
        return false;
    }

    if (message.type === "START_CAPTURE") {
        console.log("[Offscreen] Starting capture with mode:", message.mode, "streamId:", message.streamId ? "present" : "missing");
        startCapture(message.streamId, message.mode)
            .then(() => {
                console.log("[Offscreen] Capture started successfully");
                sendResponse({ status: "capturing" });
            })
            .catch((err) => {
                console.error("[Offscreen] Start capture error:", err);
                sendResponse({ error: err.message });
            });
        return true; // async response
    }

    if (message.type === "STOP_CAPTURE") {
        console.log("[Offscreen] Stopping capture");
        stopCapture(message.clicks)
            .then((result) => {
                console.log("[Offscreen] Capture stopped, result:", result);
                sendResponse(result);
            })
            .catch((err) => {
                console.error("[Offscreen] Stop capture error:", err);
                sendResponse({ error: err.message });
            });
        return true;
    }

    if (message.type === "CAPTURE_FRAME") {
        console.log("[Offscreen] Capturing frame for click");
        captureFrame(message.clickData)
            .then((frameData) => {
                console.log("[Offscreen] Frame captured");
                sendResponse({ status: "captured", frameData });
            })
            .catch((err) => {
                console.error("[Offscreen] Frame capture error:", err);
                sendResponse({ error: err.message });
            });
        return true;
    }

    return false;
});

async function startCapture(streamId, mode) {
    currentMode = mode || "tab";
    capturedFrames = []; // Reset frames for new recording
    let stream;

    if (mode === "screen") {
        // Screen capture - use getDisplayMedia to let user choose screen/window
        console.log("[Offscreen] Requesting display media for screen capture...");
        stream = await navigator.mediaDevices.getDisplayMedia({
            video: {
                displaySurface: "monitor",
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                frameRate: { ideal: 30 },
            },
            audio: true, // System audio if supported
        });
    } else {
        // Tab capture - use the provided streamId
        console.log("[Offscreen] Using tab capture with streamId...");
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                mandatory: {
                    chromeMediaSource: "tab",
                    chromeMediaSourceId: streamId,
                },
            },
            video: {
                mandatory: {
                    chromeMediaSource: "tab",
                    chromeMediaSourceId: streamId,
                    minWidth: 1280,
                    minHeight: 720,
                },
            },
        });
    }

    currentStream = stream; // Store stream for frame capture
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.start(1000); // Collect data every second
    recordingStartTime = Date.now(); // Track when recording actually started
    console.log("[Offscreen] Recording started in mode:", mode);
}

async function captureFrame(clickData) {
    if (!currentStream) {
        throw new Error("No active stream to capture frame from");
    }

    const videoTrack = currentStream.getVideoTracks()[0];
    if (!videoTrack) {
        throw new Error("No video track available");
    }

    // Create a video element to draw the frame
    const video = document.createElement("video");
    video.srcObject = new MediaStream([videoTrack]);
    video.muted = true;

    await new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve();
        };
    });

    // Wait for a frame to be ready
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Draw to canvas and convert to base64
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const frameData = canvas.toDataURL("image/png");

    // Calculate click time in seconds since recording started
    const clickTime = recordingStartTime ? (Date.now() - recordingStartTime) / 1000 : 0;

    // Store frame with click data
    const frame = {
        image: frameData,
        clickTime: parseFloat(clickTime.toFixed(2)), // Time in seconds since recording started
        clickX: clickData?.screenX,
        clickY: clickData?.screenY,
        clientX: clickData?.clientX,
        clientY: clickData?.clientY,
    };
    capturedFrames.push(frame);

    // Clean up
    video.srcObject = null;

    return frame;
}

async function stopCapture(clicks) {
    return new Promise((resolve, reject) => {
        if (!mediaRecorder) {
            resolve(null);
            return;
        }

        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordedChunks, { type: "video/webm" });

            // Stop all tracks
            mediaRecorder.stream.getTracks().forEach((track) => track.stop());
            mediaRecorder = null;
            currentStream = null;
            recordedChunks = [];

            // Get captured frames before clearing
            const frames = [...capturedFrames];
            capturedFrames = [];

            // Convert blob to base64 to send through message passing
            try {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Data = reader.result;
                    console.log("[Offscreen] Recording converted to base64, size:", base64Data.length, "frames:", frames.length);
                    resolve({
                        videoData: base64Data,
                        clicks: clicks || [],
                        frames: frames,
                        mode: currentMode,
                    });
                };
                reader.onerror = () => {
                    reject(new Error("Failed to convert recording to base64"));
                };
                reader.readAsDataURL(blob);
            } catch (err) {
                console.error("[Offscreen] Failed to convert recording:", err);
                reject(err);
            }
        };

        mediaRecorder.onerror = (err) => {
            reject(err);
        };

        mediaRecorder.stop();
    });
}
