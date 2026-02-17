/**
 * POPUP.JS - Screen Recorder UI Controller
 * 
 * This script manages the popup interface for the screen recording extension.
 * It handles:
 * - Recording controls (start/stop)
 * - UI interactions and modals
 * - Integration with backend API for video analysis
 * - Video playback and management
 * - Job polling for long-running tasks
 * 
 * Key workflows:
 * 1. User records screen/tab → saves to IndexedDB
 * 2. User submits for AI analysis → backend analyzes and generates narrations
 * 3. User renders video → backend creates final video with audio
 * 4. User downloads result → saved video is retrieved from backend
 * 
 * ============================================
 * DATA FLOW OVERVIEW
 * ============================================
 * 
 * Recording Flow:
 * START_RECORDING → background.js captures frames/audio
 *                → popup.js saves blob to IndexedDB
 *                → recording appears in "Not Analyzed" section
 * 
 * Analysis Flow:
 * analyzeRecording() → POST /analyze → backend processes video
 *                   → pollJobStatus() checks progress every 2 seconds
 *                   → on complete: recording moves to "Analyzed" section
 *                   → session_id, narrations, summary stored
 * 
 * Render Flow:
 * startRender() → POST /render/{session_id} → backend generates final video
 *              → pollRenderJobStatus() checks progress every 2 seconds
 *              → on complete: output_file stored in recording
 *              → next preview load will fetch from backend
 * 
 * Display Flow:
 * loadPreviews() → checks if output_file exists
 *               → if yes: loads from /download/{session}/{output_file}
 *               → if no: loads from local IndexedDB blob
 * 
 * ============================================
 * MODAL MANAGEMENT
 * ============================================
 * 
 * modeModal        - User selects recording mode (tab/screen)
 * nameModal        - User enters recording name
 * aiModal          - User selects voice, style, language for analysis
 * renderModal      - User selects render mode (voice-only/full)
 * videoModal       - Video player for playback
 * confirmModal     - Delete confirmation dialog
 * 
 * ============================================
 * PERSISTENT STATE (chrome.storage.local)
 * ============================================
 * 
 * pendingJob       - Active analysis job info (for resume on popup reopen)
 * pendingSession   - Completed analysis awaiting render
 * pendingRenderJob - Active render job info (for resume on popup reopen)
 * 
 * ============================================
 * BACKEND ENDPOINTS
 * ============================================
 * 
 * POST /analyze                        - Submit video + clicks for analysis
 *      Returns: {job_id, session_id}
 * 
 * GET /status/{job_id}                 - Check analysis/render job progress
 *     Returns: {status, progress, message, result}
 * 
 * POST /render/{session_id}            - Request rendering with mode
 *      Returns: {job_id, session_id, mode}
 * 
 * GET /download/{session}/{filename}   - Download rendered video
 *     Returns: FileResponse with video/mp4
 * 
 * DELETE /session/{session_id}         - Clean up session files
 */

// ============================================
// DOM ELEMENT REFERENCES
// ============================================
const stopRecordBtn = document.getElementById("stopRecordBtn");
const recordingStatus = document.getElementById("recordingStatus");
const statusText = document.getElementById("statusText");

// Recordings elements
const recordingsList = document.getElementById("recordingsList");
const analyzedList = document.getElementById("analyzedList");
const clearRecordingsBtn = document.getElementById("clearRecordingsBtn");

// Video modal elements
const videoModal = document.getElementById("videoModal");
const videoPlayer = document.getElementById("videoPlayer");
const closeVideoBtn = document.getElementById("closeVideoBtn");

// Mode selection modal elements
const modeModal = document.getElementById("modeModal");
const modeTabBtn = document.getElementById("modeTab");
const modeScreenBtn = document.getElementById("modeScreen");
const modeCancelBtn = document.getElementById("modeCancelBtn");

// Name input modal elements
const nameModal = document.getElementById("nameModal");
const recordingNameInput = document.getElementById("recordingNameInput");
const saveNameBtn = document.getElementById("saveNameBtn");

// Confirm modal elements
const confirmModal = document.getElementById("confirmModal");
const confirmTitle = document.getElementById("confirmTitle");
const confirmMessage = document.getElementById("confirmMessage");
const confirmCancelBtn = document.getElementById("confirmCancelBtn");
const confirmDeleteBtn = document.getElementById("confirmDeleteBtn");

// Success modal elements
const successModal = document.getElementById("successModal");
const successTitle = document.getElementById("successTitle");
const successMessage = document.getElementById("successMessage");
const successOkBtn = document.getElementById("successOkBtn");

// AI Analysis modal elements
const aiModal = document.getElementById("aiModal");
const aiCancelBtn = document.getElementById("aiCancelBtn");
const aiGenerateBtn = document.getElementById("aiGenerateBtn");
const voiceOptions = document.getElementById("voiceOptions");
const styleOptions = document.getElementById("styleOptions");
const aiLanguage = document.getElementById("aiLanguage");
const aiProgress = document.getElementById("aiProgress");
const aiProgressFill = document.getElementById("aiProgressFill");
const aiProgressText = document.getElementById("aiProgressText");
let currentAiRecordingId = null;
let currentAbortController = null;  // Track ongoing request for cancellation

// Render Mode Selection modal elements
const renderModal = document.getElementById("renderModal");
const renderVoiceOnlyBtn = document.getElementById("renderVoiceOnly");
const renderFullBtn = document.getElementById("renderFull");
const renderCancelBtn = document.getElementById("renderCancelBtn");
const renderProgress = document.getElementById("renderProgress");
const renderProgressFill = document.getElementById("renderProgressFill");
const renderProgressText = document.getElementById("renderProgressText");
let currentSessionData = null;  // Store analysis result for rendering
let pollingInterval = null;  // Track polling interval for job status
let renderPollingInterval = null;  // Track polling interval for render job status

// Theme toggle
const themeToggle = document.getElementById("themeToggle");

// Load saved theme
const savedTheme = localStorage.getItem("theme") || "dark";
if (savedTheme === "light") {
    document.body.classList.add("light");
    themeToggle.textContent = "☀️";
}

themeToggle.addEventListener("click", () => {
    const isLight = document.body.classList.toggle("light");
    themeToggle.textContent = isLight ? "☀️" : "🌙";
    localStorage.setItem("theme", isLight ? "light" : "dark");
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Shows a custom confirm dialog modal
 * @param {string} title - Title text for the dialog
 * @param {string} message - Message body of the dialog
 * @returns {Promise<boolean>} - True if confirmed, false if cancelled
 */
function showConfirm(title, message) {
    return new Promise((resolve) => {
        confirmTitle.textContent = title;
        confirmMessage.textContent = message;
        confirmModal.classList.add("active");

        const handleConfirm = () => {
            confirmModal.classList.remove("active");
            cleanup();
            resolve(true);
        };

        const handleCancel = () => {
            confirmModal.classList.remove("active");
            cleanup();
            resolve(false);
        };

        const cleanup = () => {
            confirmDeleteBtn.removeEventListener("click", handleConfirm);
            confirmCancelBtn.removeEventListener("click", handleCancel);
        };

        confirmDeleteBtn.addEventListener("click", handleConfirm);
        confirmCancelBtn.addEventListener("click", handleCancel);
    });
}

// Temporary storage for pending recording data
let pendingRecordingData = null;

// ============================================
// RECORDING CONTROLS
// ============================================

/**
 * Updates the UI to reflect recording state
 * @param {boolean} isRecording - Whether recording is currently active
 */
function updateRecordingUI(isRecording) {
    startRecordBtn.disabled = isRecording;
    stopRecordBtn.disabled = !isRecording;

    if (isRecording) {
        recordingStatus.classList.add("active");
        statusText.textContent = "Recording...";
        startRecordBtn.classList.add("recording");
    } else {
        recordingStatus.classList.remove("active");
        statusText.textContent = "Ready to record";
        startRecordBtn.classList.remove("recording");
    }
}

// Check initial recording state
chrome.runtime.sendMessage({ type: "GET_RECORDING_STATE" }, (response) => {
    if (chrome.runtime.lastError) return;
    updateRecordingUI(response?.isRecording || false);
});

// Show mode selection modal when clicking Start
startRecordBtn.addEventListener("click", () => {
    modeModal.classList.add("active");
});

// Cancel mode selection
modeCancelBtn.addEventListener("click", () => {
    modeModal.classList.remove("active");
});

// Start recording with selected mode
async function startRecordingWithMode(mode) {
    /**
     * Initiates recording in the specified mode (tab or screen)
     * Clears previous click data and communicates with background service worker
     * @param {string} mode - "tab" for current tab only, "screen" for full screen
     */
    modeModal.classList.remove("active");
    statusText.textContent = "Starting...";

    // Clear previous clicks
    chrome.runtime.sendMessage({ type: "CLEAR_CLICKS" });

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.runtime.sendMessage({ type: "START_RECORDING", tabId: tab.id, mode: mode }, (response) => {
        if (chrome.runtime.lastError) {
            statusText.textContent = "Error starting";
            console.error("Start recording error:", chrome.runtime.lastError);
            return;
        }
        if (response?.status === "recording") {
            updateRecordingUI(true);
        } else if (response?.error) {
            statusText.textContent = "Error: " + response.error;
        }
    });
}

// Mode option click handlers
modeTabBtn.addEventListener("click", () => startRecordingWithMode("tab"));
modeScreenBtn.addEventListener("click", () => startRecordingWithMode("screen"));

stopRecordBtn.addEventListener("click", () => {
    statusText.textContent = "Saving...";
    chrome.runtime.sendMessage({ type: "STOP_RECORDING" }, async (response) => {
        if (chrome.runtime.lastError) {
            console.error(chrome.runtime.lastError);
            statusText.textContent = "Error saving";
            return;
        }
        updateRecordingUI(false);

        // Save video data to IndexedDB if present
        if (response?.videoData) {
            try {
                // Convert base64 to blob
                const byteString = atob(response.videoData.split(",")[1]);
                const mimeString = response.videoData.split(",")[0].split(":")[1].split(";")[0];
                const ab = new ArrayBuffer(byteString.length);
                const ia = new Uint8Array(ab);
                for (let i = 0; i < byteString.length; i++) {
                    ia[i] = byteString.charCodeAt(i);
                }
                const blob = new Blob([ab], { type: mimeString });

                // Store pending data and show name modal
                pendingRecordingData = {
                    blob,
                    clicks: response.clicks || [],
                    mode: response.mode || "tab",
                    frames: response.frames || []
                };

                // Show name input modal
                recordingNameInput.value = "";
                nameModal.classList.add("active");
                recordingNameInput.focus();
                statusText.textContent = "Enter recording name...";
            } catch (err) {
                console.error("Failed to process recording:", err);
                statusText.textContent = "Save failed";
            }
        } else {
            statusText.textContent = "No video data";
        }
    });
});

// Save recording with name
saveNameBtn.addEventListener("click", saveRecordingWithName);
recordingNameInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        saveRecordingWithName();
    }
});

async function saveRecordingWithName() {
    /**
     * Saves the pending recording to IndexedDB with user-provided name
     * Clears pending data after successful save and reloads recording list
     */
    if (!pendingRecordingData) return;

    const name = recordingNameInput.value.trim() || `Recording ${Date.now()}`;
    nameModal.classList.remove("active");

    try {
        await window.RecordingsDB.saveRecording(
            pendingRecordingData.blob,
            pendingRecordingData.clicks,
            pendingRecordingData.mode,
            pendingRecordingData.frames,
            name
        );
        console.log("[Popup] Recording saved as:", name);
        statusText.textContent = "Saved!";
        pendingRecordingData = null;
        loadRecordings();
    } catch (err) {
        console.error("Failed to save recording:", err);
        statusText.textContent = "Save failed";
    }
}

// --- Recordings Management ---

function formatDate(isoString) {
    /**
     * Formats ISO date string to readable format (e.g., "Jan 15, 2:30 PM")
     * @param {string} isoString - ISO format date string
     * @returns {string} - Formatted date/time string
     */
    const d = new Date(isoString);
    return d.toLocaleDateString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function loadRecordings() {
    /**
     * Fetches all recordings from IndexedDB and renders them
     * Splits recordings into analyzed (with session_id) and non-analyzed categories
     */
    try {
        const recordings = await window.RecordingsDB.getRecordings();
        renderRecordings(recordings);
    } catch (err) {
        console.error("Failed to load recordings:", err);
        recordingsList.innerHTML = `<div class="empty-state"><p>Error loading recordings</p></div>`;
        analyzedList.innerHTML = `<div class="empty-state"><p>Error loading videos</p></div>`;
    }
}

function renderRecordings(recordings) {
    /**
     * Renders the recordings list UI
     * Splits into analyzed and non-analyzed sections
     * Sets up event listeners for all action buttons
     * @param {Array} recordings - Array of recording objects from IndexedDB
     */
    const analyzed = recordings ? recordings.filter(r => r.session_id) : [];
    const notAnalyzed = recordings ? recordings.filter(r => !r.session_id) : [];

    // Sort both by timestamp (newest first)
    const sortedAnalyzed = [...analyzed].sort((a, b) => new Date(b.analyzed_at || b.timestamp) - new Date(a.analyzed_at || a.timestamp));
    const sortedNotAnalyzed = [...notAnalyzed].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // SVG icons
    const playIcon = `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
    const sparklesIcon = `<svg viewBox="0 0 256 256" fill="currentColor"><path d="M208,144a15.78,15.78,0,0,1-10.42,14.94L168,168l-9.06,29.58a16,16,0,0,1-29.88,0L120,168l-29.58-9.06a16,16,0,0,1,0-29.88L120,120l9.06-29.58a16,16,0,0,1,29.88,0L168,120l29.58,9.06A15.78,15.78,0,0,1,208,144ZM60,80a12,12,0,0,0,4.69-1L80,73.25,95.31,79a12,12,0,0,0,9.38,0L120,73.25,135.31,79a12,12,0,0,0,9.38-22.1L128,50.75V36a12,12,0,0,0-24,0V50.75L87.31,57a12,12,0,0,0,0,22.1L104,85.25V100a12,12,0,0,0,24,0V85.25L144.69,79a12,12,0,0,0,0-22.1L128,50.75V36a12,12,0,0,0-24,0V50.75L87.31,57a12,12,0,0,0-9.38,22.1L95.31,79,80,73.25,64.69,79A12,12,0,0,0,60,80Z"/></svg>`;
    const downloadIcon = `<svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>`;
    const framesIcon = `<svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5-7l-3 3.72L9 13l-3 4h12l-4-5z"/></svg>`;
    const deleteIcon = `<svg viewBox="0 0 24 24"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>`;
    const renderIcon = `<svg viewBox="0 0 24 24"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>`;
    const uploadIcon = `<svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>`;

    // Helper to generate recording item HTML
    const createRecordingItem = (rec, isAnalyzed) => {
        const modeLabel = rec.mode === "screen" ? "Screen" : "Tab";
        const framesCount = rec.frames?.length || 0;
        const displayName = rec.name || `Recording #${rec.id}`;

        if (isAnalyzed) {
            return `
                <div class="recording-item analyzed" data-id="${rec.id}" data-session="${rec.session_id}">
                    <div class="rec-preview" data-id="${rec.id}">
                        <video muted></video>
                        <div class="preview-play">${playIcon}</div>
                    </div>
                    <div class="rec-info">
                        <div class="rec-header">
                            <span class="rec-title" title="${displayName}">${displayName}</span>
                            <span class="rec-meta analyzed-badge">✓ Analyzed</span>
                        </div>
                        <div class="rec-meta">${formatDate(rec.analyzed_at || rec.timestamp)} · ${rec.narrations?.length || 0} narrations</div>
                        <div class="rec-actions">
                            <button class="play-btn" data-id="${rec.id}" title="Play">${playIcon}</button>
                            <button class="render-btn" data-id="${rec.id}" data-session="${rec.session_id}" title="Render Video">${renderIcon}</button>
                            <button class="upload-btn" data-id="${rec.id}" data-session="${rec.session_id}" title="Upload to Server">${uploadIcon}</button>
                            <button class="download-btn" data-id="${rec.id}" title="Download Analyzed">${downloadIcon}</button>
                            <button class="delete-btn" data-id="${rec.id}" title="Delete">${deleteIcon}</button>
                        </div>
                    </div>
                </div>`;
        } else {
            return `
                <div class="recording-item" data-id="${rec.id}">
                    <div class="rec-preview" data-id="${rec.id}">
                        <video muted></video>
                        <div class="preview-play">${playIcon}</div>
                    </div>
                    <div class="rec-info">
                        <div class="rec-header">
                            <span class="rec-title" title="${displayName}">${displayName}</span>
                            <span class="rec-meta">${modeLabel}</span>
                        </div>
                        <div class="rec-meta">${formatDate(rec.timestamp)} · ${framesCount} frames</div>
                        <div class="rec-actions">
                            <button class="play-btn" data-id="${rec.id}" title="Play">${playIcon}</button>
                            <button class="ai-btn" data-id="${rec.id}" title="AI Analysis">${sparklesIcon}</button>
                            <button class="download-btn" data-id="${rec.id}" title="Download Video">${downloadIcon}</button>
                            <button class="frames-btn" data-id="${rec.id}" title="Download Frames" ${framesCount === 0 ? 'disabled' : ''}>${framesIcon}</button>
                            <button class="delete-btn" data-id="${rec.id}" title="Delete">${deleteIcon}</button>
                        </div>
                    </div>
                </div>`;
        }
    };

    // Render analyzed videos
    if (sortedAnalyzed.length === 0) {
        analyzedList.innerHTML = `<div class="empty-state"><p>No analyzed videos yet</p></div>`;
    } else {
        analyzedList.innerHTML = sortedAnalyzed.map(rec => createRecordingItem(rec, true)).join("");
    }

    // Render non-analyzed recordings
    if (sortedNotAnalyzed.length === 0) {
        recordingsList.innerHTML = `<div class="empty-state"><p>No recordings yet</p></div>`;
    } else {
        recordingsList.innerHTML = sortedNotAnalyzed.map(rec => createRecordingItem(rec, false)).join("");
    }

    // Add event listeners for both lists
    const allLists = [recordingsList, analyzedList];

    allLists.forEach(list => {
        list.querySelectorAll(".play-btn").forEach((btn) => {
            btn.addEventListener("click", () => playRecording(parseInt(btn.dataset.id)));
        });

        list.querySelectorAll(".download-btn").forEach((btn) => {
            btn.addEventListener("click", () => downloadRecording(parseInt(btn.dataset.id)));
        });

        list.querySelectorAll(".delete-btn").forEach((btn) => {
            btn.addEventListener("click", () => deleteRecording(parseInt(btn.dataset.id)));
        });

        list.querySelectorAll(".rec-preview").forEach((preview) => {
            preview.style.cursor = "pointer";
            preview.addEventListener("click", () => playRecording(parseInt(preview.dataset.id)));
        });
    });

    // Event listeners specific to non-analyzed recordings
    recordingsList.querySelectorAll(".frames-btn").forEach((btn) => {
        btn.addEventListener("click", () => downloadFrames(parseInt(btn.dataset.id)));
    });

    recordingsList.querySelectorAll(".ai-btn").forEach((btn) => {
        btn.addEventListener("click", () => analyzeRecording(parseInt(btn.dataset.id)));
    });

    // Event listeners specific to analyzed recordings (render button)
    analyzedList.querySelectorAll(".render-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const recId = parseInt(btn.dataset.id);
            const sessionId = btn.dataset.session;

            // Get the recording to load session data
            const rec = await window.RecordingsDB.getRecording(recId);
            if (rec && rec.session_id) {
                currentSessionData = {
                    session_id: rec.session_id,
                    narrations: rec.narrations,
                    summary: rec.summary
                };
                currentAiRecordingId = recId;
                renderModal.classList.add("active");
            }
        });
    });

    // Event listeners specific to analyzed recordings (upload button)
    analyzedList.querySelectorAll(".upload-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const recId = parseInt(btn.dataset.id);
            const sessionId = btn.dataset.session;
            await uploadRecording(recId, sessionId);
        });
    });

    // Load video previews for both lists
    loadPreviews([...sortedAnalyzed, ...sortedNotAnalyzed]);
}

async function loadPreviews(recordings) {
    /**
     * Loads video previews for all recordings
     * For rendered videos, fetches from backend; for others, uses local blobs
     * Seeks to 0.5s to show a representative frame
     * @param {Array} recordings - Array of recording objects
     */
    for (const rec of recordings) {
        try {
            const recording = await window.RecordingsDB.getRecording(rec.id);
            if (!recording) continue;

            // Check both lists for the preview element
            let previewEl = recordingsList.querySelector(`.rec-preview[data-id="${rec.id}"] video`);
            if (!previewEl) {
                previewEl = analyzedList.querySelector(`.rec-preview[data-id="${rec.id}"] video`);
            }
            if (!previewEl) continue;

            // For analyzed videos with rendered output, load from backend
            if (recording.output_file && recording.session_id) {
                const url = `http://localhost:8000/download/${recording.session_id}/${recording.output_file}`;
                previewEl.src = url;
                console.log(`Loading rendered video from backend: ${url}`);
            } else {
                // For non-analyzed or non-rendered videos, load from local blob
                const url = URL.createObjectURL(recording.blob);
                previewEl.src = url;
            }

            // Seek to 0.5 second to get a representative frame
            previewEl.onloadedmetadata = () => {
                previewEl.currentTime = Math.min(0.5, previewEl.duration / 2);
            };
        } catch (err) {
            console.log("Could not load preview for recording", rec.id);
        }
    }
}

async function playRecording(id) {
    /**
     * Opens video player modal and plays the specified recording
     * Loads from backend if rendered, from local blob otherwise
     * @param {number} id - Recording ID from IndexedDB
     */
    try {
        const recording = await window.RecordingsDB.getRecording(id);
        if (!recording) return;

        // For analyzed videos with rendered output, play from backend
        if (recording.output_file && recording.session_id) {
            const url = `http://localhost:8000/download/${recording.session_id}/${recording.output_file}`;
            videoPlayer.src = url;
            console.log(`Playing rendered video from backend: ${url}`);
        } else {
            // For non-analyzed or non-rendered videos, play from local blob
            const url = URL.createObjectURL(recording.blob);
            videoPlayer.src = url;
        }

        videoPlayer.onloadedmetadata = () => videoPlayer.play();
        videoModal.classList.add("active");
    } catch (err) {
        console.error("Failed to play recording:", err);
    }
}

async function downloadRecording(id) {
    /**
     * Downloads the recording or analyzed video
     * For analyzed videos with output_file, downloads the rendered video from backend
     * Otherwise downloads the original WebM blob
     * @param {number} id - Recording ID from IndexedDB
     */
    try {
        const recording = await window.RecordingsDB.getRecording(id);
        if (!recording) return;

        const safeName = (recording.name || `recording-${id}`).replace(/[^a-zA-Z0-9-_]/g, "_");
        const a = document.createElement("a");

        // For analyzed videos with rendered output, download from backend
        if (recording.output_file && recording.session_id) {
            const backendUrl = `http://localhost:8000/download/${recording.session_id}/${recording.output_file}`;
            a.href = backendUrl;
            a.download = recording.output_file;
            console.log(`Downloading analyzed video from backend: ${backendUrl}`);
        } else {
            // For non-analyzed or non-rendered videos, download from local blob
            const url = URL.createObjectURL(recording.blob);
            a.href = url;
            a.download = `${safeName}.webm`;
        }

        a.click();
        if (!recording.output_file || !recording.session_id) {
            URL.revokeObjectURL(a.href);
        }
    } catch (err) {
        console.error("Failed to download recording:", err);
    }
}

async function uploadRecording(id, sessionId) {
    /**
     * Uploads the analyzed video as a kit to the central website
     * via the backend's /upload/{session_id} endpoint.
     * 
     * @param {number} id - Recording ID from IndexedDB
     * @param {string} sessionId - Session ID from backend analysis
     */

    const uploadBtn = document.querySelector(`.upload-btn[data-id="${id}"]`);
    const originalHTML = uploadBtn ? uploadBtn.innerHTML : '';

    try {
        const recording = await window.RecordingsDB.getRecording(id);
        if (!recording || !recording.session_id) {
            console.error("No analyzed session available to upload");
            successTitle.textContent = "Upload Failed";
            successMessage.textContent = "Please analyze the video first before uploading.";
            document.querySelector(".success-icon").textContent = "✗";
            successModal.classList.add("active");
            return;
        }

        // Show loading state on the button
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = `<svg class="spin" viewBox="0 0 24 24"><path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>`;
        }

        console.log(`[UPLOAD] Uploading recording ${id}, session ${sessionId}, frames: ${recording.frames?.length || 0}`);

        const response = await fetch(`http://localhost:8000/upload/${sessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frames: recording.frames || []
            })
        });

        const result = await response.json();

        if (result.status === "ok") {
            console.log("[UPLOAD] Success:", result);

            // Update recording in IndexedDB with upload status
            await window.RecordingsDB.updateRecording(id, {
                uploaded_at: new Date().toISOString(),
                upload_status: 'completed'
            });

            // Show success modal
            document.querySelector(".success-icon").textContent = "✓";
            successTitle.textContent = "Kit Uploaded!";
            successMessage.textContent = result.message || "Your kit has been uploaded to the central website.";
            successModal.classList.add("active");

            // Refresh recordings list
            loadRecordings();
        } else {
            console.error("[UPLOAD] Failed:", result);
            document.querySelector(".success-icon").textContent = "✗";
            successTitle.textContent = "Upload Failed";
            successMessage.textContent = result.message || "Failed to upload kit.";
            successModal.classList.add("active");
        }

    } catch (err) {
        console.error("Failed to upload recording:", err);
        document.querySelector(".success-icon").textContent = "✗";
        successTitle.textContent = "Upload Error";
        successMessage.textContent = "Could not connect to the server. Is the backend running?";
        successModal.classList.add("active");
    } finally {
        // Restore button state
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = originalHTML;
        }
    }
}

async function downloadFrames(id) {
    /**
     * Downloads all captured frames as PNG images + JSON metadata file
     * Frames are snapshots at each click moment during recording
     * @param {number} id - Recording ID from IndexedDB
     */
    try {
        const recording = await window.RecordingsDB.getRecording(id);
        if (!recording || !recording.frames || recording.frames.length === 0) {
            console.log("No frames to download");
            return;
        }

        const safeName = (recording.name || `recording-${id}`).replace(/[^a-zA-Z0-9-_]/g, "_");

        // Download each frame as a PNG file
        for (let i = 0; i < recording.frames.length; i++) {
            const frame = recording.frames[i];
            const a = document.createElement("a");
            a.href = frame.image;
            const clickTimeStr = (frame.clickTime || 0).toFixed(2).replace(".", "s");
            a.download = `${safeName}_frame_${i + 1}_${clickTimeStr}.png`;

            // Small delay between downloads to avoid browser blocking
            await new Promise(resolve => setTimeout(resolve, 100));
            a.click();
        }

        // Also download a JSON file with click coordinates for each frame
        const framesData = recording.frames.map((frame, i) => ({
            frameNumber: i + 1,
            clickTime: frame.clickTime,
            clickScreenX: frame.clickX,
            clickScreenY: frame.clickY,
            clickClientX: frame.clientX,
            clickClientY: frame.clientY,
        }));

        const jsonBlob = new Blob([JSON.stringify(framesData, null, 2)], { type: "application/json" });
        const jsonUrl = URL.createObjectURL(jsonBlob);
        const jsonLink = document.createElement("a");
        jsonLink.href = jsonUrl;
        jsonLink.download = `${safeName}_frames_data.json`;
        jsonLink.click();
        URL.revokeObjectURL(jsonUrl);

        console.log(`Downloaded ${recording.frames.length} frames for recording ${id}`);
    } catch (err) {
        console.error("Failed to download frames:", err);
    }
}

async function deleteRecording(id) {
    /**
     * Deletes a recording from IndexedDB after user confirmation
     * @param {number} id - Recording ID to delete
     */
    const confirmed = await showConfirm("Delete Recording?", "This action cannot be undone.");
    if (!confirmed) {
        return;
    }
    try {
        await window.RecordingsDB.deleteRecording(id);
        loadRecordings();
    } catch (err) {
        console.error("Failed to delete recording:", err);
    }
}

async function analyzeRecording(id) {
    /**
     * Opens AI analysis modal for the specified recording
     * User can select voice, narration style, and language before submitting
     * @param {number} id - Recording ID to analyze
     */
    currentAiRecordingId = id;
    aiGenerateBtn.disabled = false;
    aiProgress.style.display = "none";
    aiProgressFill.style.width = "0%";
    aiProgressText.textContent = "Configure your tutorial settings...";
    aiModal.classList.add("active");
}

// AI Modal option selection
function setupOptionSelection(container) {
    container.addEventListener("click", (e) => {
        const option = e.target.closest(".ai-option");
        if (!option) return;
        container.querySelectorAll(".ai-option").forEach(o => o.classList.remove("selected"));
        option.classList.add("selected");
    });
}
setupOptionSelection(voiceOptions);
setupOptionSelection(styleOptions);

// AI Cancel
aiCancelBtn.addEventListener("click", async () => {
    // Stop polling if active
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log("Stopped polling");
    }

    // Abort ongoing request if any
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        console.log("Analysis cancelled by user");
    }

    // Clear pending job from storage
    await chrome.storage.local.remove("pendingJob");

    aiModal.classList.remove("active");
    aiProgress.style.display = "none";
    aiProgressFill.style.width = "0%";
    aiGenerateBtn.disabled = false;
    currentAiRecordingId = null;
});

// AI Generate
aiGenerateBtn.addEventListener("click", async () => {
    if (!currentAiRecordingId) return;

    const voice = voiceOptions.querySelector(".ai-option.selected")?.dataset.value || "Sarah";
    const style = styleOptions.querySelector(".ai-option.selected")?.dataset.value || "professional";
    const language = aiLanguage.value;

    try {
        const rec = await window.RecordingsDB.getRecording(currentAiRecordingId);
        if (!rec) {
            console.error("Recording not found");
            return;
        }

        console.log("Submitting analysis job:", { voice, style, language, recordingId: currentAiRecordingId });

        // Show progress and disable button
        aiProgress.style.display = "block";
        aiProgressFill.style.width = "0%";
        aiProgressText.textContent = "Submitting job...";
        aiGenerateBtn.disabled = true;

        // Send to backend API - get job_id immediately
        const formData = new FormData();
        const videoFilename = rec.name ? (rec.name.endsWith('.webm') ? rec.name : `${rec.name}.webm`) : "recording.webm";
        formData.append("video", rec.blob, videoFilename);
        formData.append("clicks", JSON.stringify(rec.clicks || []));
        formData.append("voice", voice);
        formData.append("style", style);
        formData.append("language", language);

        const response = await fetch("http://localhost:8000/analyze", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (result.status === "ok" && result.job_id) {
            console.log("Job submitted:", result.job_id);

            // Save job info to chrome.storage immediately
            await chrome.storage.local.set({
                pendingJob: {
                    job_id: result.job_id,
                    session_id: result.session_id,
                    recording_id: currentAiRecordingId
                }
            });

            // Start polling for job status
            startJobPolling(result.job_id);
        } else {
            console.error("Backend error:", result);
            alert(`Error: ${result.message || "Failed to submit job"}`);
            aiProgress.style.display = "none";
            aiGenerateBtn.disabled = false;
        }

    } catch (err) {
        console.error("Failed to submit analysis job:", err);
        aiProgressText.textContent = "Error: Failed to connect to backend";
        alert(`Failed to connect to backend: ${err.message}`);
        aiProgress.style.display = "none";
        aiGenerateBtn.disabled = false;
    }
});

// Poll job status every 2 seconds
function startJobPolling(jobId) {
    /**
     * Initiates polling for AI analysis job status
     * Polls backend every 2 seconds until job completes or errors
     * @param {string} jobId - Backend job ID returned from /analyze endpoint
     */

    // Clear any existing polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }

    // Set up interval FIRST so immediate poll can clear it if job is already complete
    pollingInterval = setInterval(() => {
        pollJobStatus(jobId);
    }, 2000);

    // Then poll immediately
    pollJobStatus(jobId);
}

async function pollJobStatus(jobId) {
    /**
     * Polls the backend for current AI analysis job status
     * Updates UI progress bar and handles job completion/error states
     * When job completes, saves session data and opens render mode selection
     * @param {string} jobId - Backend job ID to check
     */
    try {
        const response = await fetch(`http://localhost:8000/status/${jobId}`);
        const status = await response.json();

        console.log(`Job ${jobId} status:`, status.status, `${status.progress}%`, status.message);

        // Update progress bar
        aiProgressFill.style.width = `${status.progress}%`;
        aiProgressText.textContent = status.message || "Processing...";

        if (status.status === "complete") {
            // Stop polling
            clearInterval(pollingInterval);
            pollingInterval = null;

            console.log("Analysis complete:", status.result);

            // Store session data for rendering
            currentSessionData = status.result;

            // Save session_id to the recording in IndexedDB
            if (currentAiRecordingId && status.result.session_id) {
                try {
                    await window.RecordingsDB.updateRecording(currentAiRecordingId, {
                        session_id: status.result.session_id,
                        narrations: status.result.narrations,
                        summary: status.result.summary,
                        analyzed_at: new Date().toISOString()
                    });
                    console.log("Recording updated with session_id:", status.result.session_id);
                    // Refresh the recordings list to show in analyzed section
                    loadRecordings();
                } catch (err) {
                    console.error("Failed to update recording with session info:", err);
                }
            }

            // Save to chrome.storage to persist across popup reopens
            await chrome.storage.local.set({ pendingSession: status.result });

            // Clear pending job
            await chrome.storage.local.remove("pendingJob");
            console.log("Job complete, session saved");

            // Wait a moment to show complete state
            await new Promise(resolve => setTimeout(resolve, 500));

            // Close AI modal and open render mode selection
            aiModal.classList.remove("active");
            renderModal.classList.add("active");

            // Reset UI
            setTimeout(() => {
                aiProgress.style.display = "none";
                aiProgressFill.style.width = "0%";
                aiGenerateBtn.disabled = false;
            }, 500);

        } else if (status.status === "error") {
            // Stop polling
            clearInterval(pollingInterval);
            pollingInterval = null;

            console.error("Job failed:", status.error);
            aiProgressText.textContent = `Error: ${status.message}`;
            alert(`Analysis failed: ${status.message}`);

            // Clear pending job
            await chrome.storage.local.remove("pendingJob");

            // Reset UI
            setTimeout(() => {
                aiProgress.style.display = "none";
                aiProgressFill.style.width = "0%";
                aiGenerateBtn.disabled = false;
            }, 2000);
        }
        // Otherwise keep polling (status: queued, saving, analyzing, generating_tts, finalizing)

    } catch (err) {
        console.error("Failed to poll job status:", err);
        // Don't stop polling on network errors - backend might be temporarily unavailable
    }
}

// Render Mode Selection Handlers

// Voice-only mode
renderVoiceOnlyBtn.addEventListener("click", async () => {
    /**
     * Initiates "voice-only" render mode
     * Video freezes at each click point while narration plays
     */
    if (!currentSessionData) return;
    await startRender("voice-only");
});

// Full annotations mode
renderFullBtn.addEventListener("click", async () => {
    if (!currentSessionData) return;
    await startRender("full");
});

// Cancel render mode selection
renderCancelBtn.addEventListener("click", async () => {
    renderModal.classList.remove("active");
    currentSessionData = null;
    // Clear pending session from storage
    await chrome.storage.local.remove("pendingSession");
    console.log("Pending session cleared");
});

// Start rendering with selected mode
async function startRender(mode) {
    /**
     * Submits a render job to the backend
     * Shows progress UI and starts polling for job status
     * @param {string} mode - "voice-only" or "full" (full not yet implemented)
     */
    try {
        // Show progress
        renderProgress.style.display = "block";
        renderProgressFill.style.width = "0%";
        renderProgressText.textContent = `Preparing ${mode === "voice-only" ? "voice overlay" : "full annotations"}...`;

        // Disable option buttons
        renderVoiceOnlyBtn.style.pointerEvents = "none";
        renderFullBtn.style.pointerEvents = "none";
        renderVoiceOnlyBtn.style.opacity = "0.5";
        renderFullBtn.style.opacity = "0.5";

        // Call backend render endpoint
        const formData = new FormData();
        formData.append("mode", mode);

        const response = await fetch(`http://localhost:8000/render/${currentSessionData.session_id}`, {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (result.status === "ok" && result.job_id) {
            console.log("Render job submitted:", result.job_id);

            // Save render job info to chrome.storage immediately
            await chrome.storage.local.set({
                pendingRenderJob: {
                    job_id: result.job_id,
                    session_id: result.session_id,
                    mode: mode,
                    recording_id: currentAiRecordingId
                }
            });

            // Start polling for render job status
            startRenderJobPolling(result.job_id, mode);
        } else {
            console.error("Render error:", result);
            alert(`Render failed: ${result.message || "Unknown error occurred"}`);
            renderProgress.style.display = "none";
            renderVoiceOnlyBtn.style.pointerEvents = "";
            renderFullBtn.style.pointerEvents = "";
            renderVoiceOnlyBtn.style.opacity = "";
            renderFullBtn.style.opacity = "";
        }

    } catch (err) {
        console.error("Failed to start render:", err);
        renderProgressText.textContent = "Error: Failed to start render";
        alert(`Failed to start render: ${err.message}`);
        renderProgress.style.display = "none";
        renderVoiceOnlyBtn.style.pointerEvents = "";
        renderFullBtn.style.pointerEvents = "";
        renderVoiceOnlyBtn.style.opacity = "";
        renderFullBtn.style.opacity = "";
    }
}

// Poll render job status every 2 seconds
function startRenderJobPolling(jobId, mode) {
    /**
     * Initiates polling for video render job status
     * Polls backend every 2 seconds until render completes or errors
     * @param {string} jobId - Backend render job ID
     * @param {string} mode - Render mode for display purposes
     */

    // Clear any existing polling
    if (renderPollingInterval) {
        clearInterval(renderPollingInterval);
    }

    // Set up interval FIRST so immediate poll can clear it if job is already complete
    renderPollingInterval = setInterval(() => {
        pollRenderJobStatus(jobId, mode);
    }, 2000);

    // Then poll immediately
    pollRenderJobStatus(jobId, mode);
}

async function pollRenderJobStatus(jobId, mode) {
    /**
     * Polls the backend for current video render job status
     * Updates progress bar and handles job completion
     * When render completes, updates recording with output_file for future playback
     * @param {string} jobId - Backend render job ID
     * @param {string} mode - Render mode ("voice-only" or "full")
     */
    try {
        const response = await fetch(`http://localhost:8000/status/${jobId}`);
        const status = await response.json();

        console.log(`Render job ${jobId} status:`, status.status, `${status.progress}%`, status.message);

        // Update progress bar
        renderProgressFill.style.width = `${status.progress}%`;
        renderProgressText.textContent = status.message || "Rendering...";

        if (status.status === "complete") {
            // Stop polling
            clearInterval(renderPollingInterval);
            renderPollingInterval = null;

            console.log("Render complete:", status.result);

            // Update the recording with the rendered output file
            if (currentAiRecordingId && status.result && status.result.output_file) {
                try {
                    await window.RecordingsDB.updateRecording(currentAiRecordingId, {
                        output_file: status.result.output_file,
                        rendered_mode: mode,
                        rendered_at: new Date().toISOString()
                    });
                    console.log("Recording updated with rendered output_file:", status.result.output_file);
                    // Refresh the recordings list to update the preview
                    loadRecordings();
                } catch (err) {
                    console.error("Failed to update recording with output_file:", err);
                }
            }

            // Clear pending render job
            await chrome.storage.local.remove("pendingRenderJob");

            // Wait a moment to show complete state
            await new Promise(resolve => setTimeout(resolve, 500));

            // Close render modal
            renderModal.classList.remove("active");

            // Show success modal instead of alert
            successModal.classList.add("active");

            // Reset UI
            setTimeout(async () => {
                renderProgress.style.display = "none";
                renderProgressFill.style.width = "0%";
                renderVoiceOnlyBtn.style.pointerEvents = "";
                renderFullBtn.style.pointerEvents = "";
                renderVoiceOnlyBtn.style.opacity = "";
                renderFullBtn.style.opacity = "";
                currentSessionData = null;
                // Clear pending session from storage after render completes
                await chrome.storage.local.remove("pendingSession");
                console.log("Pending session cleared after render");
            }, 1000);

        } else if (status.status === "error") {
            // Stop polling
            clearInterval(renderPollingInterval);
            renderPollingInterval = null;

            console.error("Render failed:", status.error);
            renderProgressText.textContent = `Error: ${status.message}`;
            alert(`Render failed: ${status.message}`);

            // Clear pending render job
            await chrome.storage.local.remove("pendingRenderJob");

            // Reset UI
            setTimeout(() => {
                renderProgress.style.display = "none";
                renderProgressFill.style.width = "0%";
                renderVoiceOnlyBtn.style.pointerEvents = "";
                renderFullBtn.style.pointerEvents = "";
                renderVoiceOnlyBtn.style.opacity = "";
                renderFullBtn.style.opacity = "";
            }, 2000);
        }
        // Otherwise keep polling (status: queued, rendering)

    } catch (err) {
        console.error("Failed to poll render job status:", err);
        // Don't stop polling on network errors - backend might be temporarily unavailable
    }
}

// Close video modal
closeVideoBtn.addEventListener("click", () => {
    videoModal.classList.remove("active");
    videoPlayer.pause();
    videoPlayer.src = "";
});

// Handle success modal OK button
successOkBtn.addEventListener("click", async () => {
    successModal.classList.remove("active");
    // Clear pending session from storage after user acknowledges success
    await chrome.storage.local.remove("pendingSession");
    console.log("Pending session cleared after user confirmation");
});

// Clear all recordings
clearRecordingsBtn.addEventListener("click", async () => {
    const confirmed = await showConfirm("Delete All Recordings?", "All recordings will be permanently deleted.");
    if (confirmed) {
        try {
            await window.RecordingsDB.clearAllRecordings();
            loadRecordings();
        } catch (err) {
            console.error("Failed to clear recordings:", err);
        }
    }
});

// Check for pending session or job on popup open
async function restorePendingSession() {
    /**
     * Restores incomplete jobs/sessions when popup is reopened
     * Checks for 3 types of pending tasks:
     * 1. Ongoing render job - resumes polling
     * 2. Completed analysis awaiting render - shows render modal
     * 3. Ongoing analysis job - resumes polling
     * Allows users to continue where they left off if popup was closed
     */
    try {
        const storage = await chrome.storage.local.get(["pendingSession", "pendingJob", "pendingRenderJob"]);

        // Check for ongoing render job that needs to resume polling
        if (storage.pendingRenderJob) {
            console.log("Restoring pending render job:", storage.pendingRenderJob);

            // Show render modal with progress
            renderModal.classList.add("active");
            renderProgress.style.display = "block";
            renderProgressFill.style.width = "0%";
            renderProgressText.textContent = "Reconnecting to render job...";
            renderVoiceOnlyBtn.style.pointerEvents = "none";
            renderFullBtn.style.pointerEvents = "none";
            renderVoiceOnlyBtn.style.opacity = "0.5";
            renderFullBtn.style.opacity = "0.5";

            // Resume polling
            startRenderJobPolling(storage.pendingRenderJob.job_id, storage.pendingRenderJob.mode);
            return;
        }

        // Check for completed session awaiting render mode selection
        if (storage.pendingSession) {
            console.log("Restoring pending session:", storage.pendingSession);
            currentSessionData = storage.pendingSession;
            renderModal.classList.add("active");
            return;
        }

        // Check for ongoing analysis job that needs to resume polling
        if (storage.pendingJob) {
            console.log("Restoring pending job:", storage.pendingJob);
            currentAiRecordingId = storage.pendingJob.recording_id;

            // Show AI modal with progress
            aiModal.classList.add("active");
            aiProgress.style.display = "block";
            aiProgressFill.style.width = "0%";
            aiProgressText.textContent = "Reconnecting to job...";
            aiGenerateBtn.disabled = true;

            // Resume polling
            startJobPolling(storage.pendingJob.job_id);
        }
    } catch (err) {
        console.error("Failed to restore pending session/job:", err);
    }
}

// Load recordings on popup open
loadRecordings();
restorePendingSession();