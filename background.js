// Background service worker: stores click events and manages recording

const MAX_CLICKS = 100; // Keep last 100 clicks
let isRecording = false;
let recordingTabId = null;

// Inject content script into all existing tabs when extension is installed/updated
chrome.runtime.onInstalled.addListener(async () => {
    // Initialize recording state
    await chrome.storage.local.set({ isRecording: false });

    const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
    for (const tab of tabs) {
        try {
            await chrome.scripting.insertCSS({
                target: { tabId: tab.id, allFrames: true },
                files: ["content.css"],
            });
            await chrome.scripting.executeScript({
                target: { tabId: tab.id, allFrames: true },
                files: ["content.js"],
            });
        } catch (err) {
            // Some tabs (e.g. chrome:// pages) can't be scripted — skip them
            console.log(`Could not inject into tab ${tab.id}: ${err.message}`);
        }
    }
    console.log("[Click Detector] Injected into all existing tabs.");
});

// Create offscreen document for recording
async function setupOffscreenDocument() {
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ["OFFSCREEN_DOCUMENT"],
    });

    if (existingContexts.length > 0) {
        return; // Already exists
    }

    await chrome.offscreen.createDocument({
        url: "offscreen.html",
        reasons: ["USER_MEDIA", "DISPLAY_MEDIA"],
        justification: "Recording tab or screen capture for click detector",
    });

    // Wait for offscreen document to be fully ready
    await new Promise((resolve) => setTimeout(resolve, 300));
}

async function closeOffscreenDocument() {
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ["OFFSCREEN_DOCUMENT"],
    });

    if (existingContexts.length > 0) {
        await chrome.offscreen.closeDocument();
    }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // Ignore messages targeted at offscreen document
    if (message.target === "offscreen") {
        return false;
    }

    if (message.type === "CLICK_EVENT") {
        // Only store clicks if recording is active
        chrome.storage.local.get({ isRecording: false }, async (state) => {
            if (!state.isRecording) {
                sendResponse({ status: "ignored", reason: "not recording" });
                return;
            }

            // Capture frame for this click
            try {
                await chrome.runtime.sendMessage({
                    type: "CAPTURE_FRAME",
                    target: "offscreen",
                    clickData: message.data,
                });
                console.log("[Background] Frame captured for click");
            } catch (err) {
                console.log("[Background] Could not capture frame:", err.message);
            }

            // Retrieve existing clicks, append, and trim
            chrome.storage.local.get({ clicks: [] }, (result) => {
                const clicks = result.clicks;
                clicks.push(message.data);

                // Keep only the last MAX_CLICKS entries
                if (clicks.length > MAX_CLICKS) {
                    clicks.splice(0, clicks.length - MAX_CLICKS);
                }

                chrome.storage.local.set({ clicks }, () => {
                    sendResponse({ status: "stored", total: clicks.length });
                });
            });
        });

        // Return true to indicate async sendResponse
        return true;
    }

    if (message.type === "GET_CLICKS") {
        chrome.storage.local.get({ clicks: [] }, (result) => {
            sendResponse({ clicks: result.clicks });
        });
        return true;
    }

    if (message.type === "CLEAR_CLICKS") {
        chrome.storage.local.set({ clicks: [] }, () => {
            sendResponse({ status: "cleared" });
        });
        return true;
    }

    if (message.type === "GET_RECORDING_STATE") {
        chrome.storage.local.get({ isRecording: false }, (result) => {
            sendResponse({ isRecording: result.isRecording });
        });
        return true;
    }

    if (message.type === "START_RECORDING") {
        handleStartRecording(message.tabId, message.mode || "tab")
            .then((result) => sendResponse(result))
            .catch((err) => sendResponse({ error: err.message }));
        return true;
    }

    if (message.type === "STOP_RECORDING") {
        handleStopRecording()
            .then((result) => sendResponse(result))
            .catch((err) => sendResponse({ error: err.message }));
        return true;
    }
});

async function handleStartRecording(tabId, mode) {
    try {
        console.log("[Background] Starting recording for tab:", tabId, "mode:", mode);

        // Clear previous clicks
        await chrome.storage.local.set({ clicks: [], isRecording: true, recordingMode: mode });
        isRecording = true;
        recordingTabId = tabId;

        // Set up offscreen document
        console.log("[Background] Setting up offscreen document...");
        await setupOffscreenDocument();

        let streamId = null;

        // For tab mode, get tab capture stream ID
        if (mode === "tab") {
            console.log("[Background] Getting tab media stream ID...");
            try {
                streamId = await chrome.tabCapture.getMediaStreamId({
                    targetTabId: tabId,
                });
            } catch (tabCaptureErr) {
                throw new Error("Tab capture failed: " + tabCaptureErr.message);
            }
            console.log("[Background] Stream ID obtained:", streamId ? "yes" : "no");
        }
        // For screen mode, streamId will be null and offscreen will use getDisplayMedia

        // Tell offscreen document to start capturing via a targeted message
        // Find the offscreen document and send message to it
        const offscreenContexts = await chrome.runtime.getContexts({
            contextTypes: ["OFFSCREEN_DOCUMENT"],
        });

        console.log("[Background] Offscreen contexts:", offscreenContexts.length);

        if (offscreenContexts.length === 0) {
            throw new Error("Offscreen document not found");
        }

        // Send message to offscreen document using its documentId
        console.log("[Background] Sending START_CAPTURE to offscreen...");
        const response = await chrome.runtime.sendMessage({
            type: "START_CAPTURE",
            streamId: streamId,
            mode: mode,
            target: "offscreen",
        });

        console.log("[Background] Offscreen response:", response);

        if (response?.error) {
            throw new Error(response.error);
        }

        // Notify all tabs that recording has started
        const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
        for (const tab of tabs) {
            try {
                await chrome.tabs.sendMessage(tab.id, { type: "RECORDING_STATE", isRecording: true });
            } catch (e) {
                // Tab might not have content script, ignore
            }
        }

        console.log("[Background] Recording started successfully");
        return { status: "recording" };
    } catch (err) {
        await chrome.storage.local.set({ isRecording: false });
        isRecording = false;
        console.error("[Background] Failed to start recording:", err);
        throw err;
    }
}

async function handleStopRecording() {
    try {
        isRecording = false;
        await chrome.storage.local.set({ isRecording: false });

        // Get the recorded clicks
        const clicksResult = await chrome.storage.local.get({ clicks: [] });
        const clicks = clicksResult.clicks;

        // Tell offscreen document to stop capturing and return video data
        let videoData = null;
        let recordingClicks = [];
        let frames = [];
        let mode = "tab";
        try {
            const response = await chrome.runtime.sendMessage({
                type: "STOP_CAPTURE",
                target: "offscreen",
                clicks: clicks,
            });
            videoData = response?.videoData || null;
            recordingClicks = response?.clicks || [];
            frames = response?.frames || [];
            mode = response?.mode || "tab";
            console.log("[Background] Received video data from offscreen, size:", videoData?.length || 0, "frames:", frames.length);
        } catch (e) {
            console.log("[Background] No active capture to stop:", e);
        }

        // Close offscreen document
        await closeOffscreenDocument();

        // Notify all tabs that recording has stopped
        const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
        for (const tab of tabs) {
            try {
                await chrome.tabs.sendMessage(tab.id, { type: "RECORDING_STATE", isRecording: false });
            } catch (e) {
                // Tab might not have content script, ignore
            }
        }

        console.log("[Background] Recording stopped");
        return { status: "stopped", videoData, clicks: recordingClicks, frames, mode };
    } catch (err) {
        console.error("[Background] Failed to stop recording:", err);
        throw err;
    }
}
