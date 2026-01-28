// background.js - Handles network requests

const BACKEND_URL = "http://localhost:8005";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // 1. Single Link Ingestion
    if (message.type === 'INGEST_LINK') {
        postData('/extension/ingest', { link: message.link })
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));
        return true; 
    }
    
    // 2. User History Bulk Ingestion
    if (message.type === 'INGEST_USER_HISTORY') {
        postData('/extension/ingest_user_history', message.payload)
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));
        return true;
    }

    // 3. Trigger Agentic Analysis
    if (message.type === 'TRIGGER_USER_ANALYSIS') {
        postData('/analyze/user_context', message.payload)
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));
        return true;
    }
    
    // 4. Manual Labeling
    if (message.type === 'SAVE_MANUAL') {
        postData('/extension/save_manual', message.payload)
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));
        return true;
    }

    // 5. Save Comments
    if (message.type === 'SAVE_COMMENTS') {
        postData('/extension/save_comments', message.payload)
            .then(data => sendResponse({success: true, data}))
            .catch(err => sendResponse({success: false, error: err.toString()}));
        return true;
    }
});

async function postData(endpoint, body) {
    try {
        const response = await fetch(`${BACKEND_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Request to ${endpoint} failed:`, error);
        throw error;
    }
}