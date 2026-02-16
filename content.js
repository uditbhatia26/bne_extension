// Content script: runs on every web page to detect clicks

// Guard against double injection (manifest + programmatic)
if (window.__clickDetectorLoaded) {
    // Already running in this page — skip
} else {
    window.__clickDetectorLoaded = true;

    let isRecording = false;

    // Check initial recording state
    chrome.storage.local.get({ isRecording: false }, (result) => {
        isRecording = result.isRecording;
        console.log(`[Click Detector] Initial recording state: ${isRecording}`);
    });

    // Listen for recording state changes from background
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.type === "RECORDING_STATE") {
            isRecording = message.isRecording;
            console.log(`[Click Detector] Recording state changed via message: ${isRecording}`);
        }
    });

    // Also listen for storage changes as a backup
    chrome.storage.onChanged.addListener((changes, areaName) => {
        if (areaName === "local" && changes.isRecording !== undefined) {
            isRecording = changes.isRecording.newValue;
            console.log(`[Click Detector] Recording state changed via storage: ${isRecording}`);
        }
    });

    function createRipple(clientX, clientY) {
        const ripple = document.createElement("div");
        ripple.className = "click-detector-ripple";
        ripple.style.left = clientX + "px";
        ripple.style.top = clientY + "px";
        document.body.appendChild(ripple);

        ripple.addEventListener("animationend", () => {
            ripple.remove();
        });
    }

    // Listen for all clicks on the page
    document.addEventListener(
        "click",
        (event) => {
            // Only process clicks when recording is active
            if (!isRecording) {
                return;
            }

            const clickData = {
                screenX: event.screenX,
                screenY: event.screenY,
                clientX: event.clientX,
                clientY: event.clientY,
                pageX: event.pageX,
                pageY: event.pageY,
                target: event.target.tagName,
                targetId: event.target.id || "",
                targetClass: event.target.className || "",
                timestamp: new Date().toISOString(),
                url: window.location.href,
                scrollX: window.scrollX,
                scrollY: window.scrollY,
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio,
                zoomLevel: window.devicePixelRatio / (window.outerWidth / window.innerWidth)
            };

            console.log(
                `[Click Detector] Screen: (${clickData.screenX}, ${clickData.screenY}) | Client: (${clickData.clientX}, ${clickData.clientY}) | Page: (${clickData.pageX}, ${clickData.pageY})`
            );

            // Show ripple animation
            createRipple(event.clientX, event.clientY);

            // Send click data to background script for storage
            chrome.runtime.sendMessage(
                { type: "CLICK_EVENT", data: clickData },
                (response) => {
                    if (chrome.runtime.lastError) {
                        // Extension context may have been invalidated, ignore silently
                    }
                }
            );
        },
        true
    );

    console.log("[Click Detector] Content script loaded and listening for clicks.");

} // end of double-injection guard
