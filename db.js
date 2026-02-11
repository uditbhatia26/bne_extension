// IndexedDB wrapper for storing recordings

const DB_NAME = "ClickDetectorDB";
const DB_VERSION = 1;
const STORE_NAME = "recordings";

let db = null;

async function openDB() {
    if (db) return db;

    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);

        request.onsuccess = () => {
            db = request.result;
            resolve(db);
        };

        request.onupgradeneeded = (event) => {
            const database = event.target.result;
            if (!database.objectStoreNames.contains(STORE_NAME)) {
                const store = database.createObjectStore(STORE_NAME, {
                    keyPath: "id",
                    autoIncrement: true,
                });
                store.createIndex("timestamp", "timestamp", { unique: false });
            }
        };
    });
}

async function saveRecording(blob, clicks, mode, frames, name) {
    const database = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);

        const recording = {
            blob: blob,
            clicks: clicks,
            frames: frames || [], // Screenshot frames captured on clicks
            timestamp: new Date().toISOString(),
            duration: 0, // Will be updated when video metadata loads
            mode: mode || "tab", // "tab" or "screen"
            name: name || "", // User-provided name
        };

        const request = store.add(recording);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function getRecordings() {
    const database = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readonly");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.getAll();

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function getRecording(id) {
    const database = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readonly");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.get(id);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function deleteRecording(id) {
    const database = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.delete(id);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function clearAllRecordings() {
    const database = await openDB();
    return new Promise((resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);
        const request = store.clear();

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function updateRecording(id, updates) {
    const database = await openDB();
    return new Promise(async (resolve, reject) => {
        const transaction = database.transaction([STORE_NAME], "readwrite");
        const store = transaction.objectStore(STORE_NAME);

        // Get existing recording
        const getRequest = store.get(id);
        getRequest.onsuccess = () => {
            const recording = getRequest.result;
            if (!recording) {
                reject(new Error("Recording not found"));
                return;
            }

            // Merge updates
            const updated = { ...recording, ...updates };
            const putRequest = store.put(updated);
            putRequest.onsuccess = () => resolve(updated);
            putRequest.onerror = () => reject(putRequest.error);
        };
        getRequest.onerror = () => reject(getRequest.error);
    });
}

// Export for use in other scripts
if (typeof window !== "undefined") {
    window.RecordingsDB = {
        saveRecording,
        getRecordings,
        getRecording,
        deleteRecording,
        clearAllRecordings,
        updateRecording,
    };
}
