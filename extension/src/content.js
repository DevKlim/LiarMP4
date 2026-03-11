// content.js - Injects buttons, Side Panel, and Handles Scraping
console.log("[vChat] Extension loaded.");

let debounceTimer = null;
let sidePanel = null;
let panelContent = null;
let toggleBtn = null;
let globalDropdown = null;

// State
let currentLink = "";
let currentCaption = "";
let currentPlatform = "";

// ============================================================================
// MODULE: Global Dropdown & Context Menu
// ============================================================================
function initGlobalDropdown() {
    if (document.getElementById('vchat-global-dropdown')) return;
    globalDropdown = document.createElement('div');
    globalDropdown.id = 'vchat-global-dropdown';
    globalDropdown.className = 'vchat-dropdown-menu';
    document.body.appendChild(globalDropdown);
    
    // Close on outside click
    window.addEventListener('click', () => { 
        if (globalDropdown) globalDropdown.style.display = 'none'; 
    });
    
    // Close on scroll (since it's position: fixed, scrolling away breaks alignment)
    window.addEventListener('scroll', () => { 
        if (globalDropdown) globalDropdown.style.display = 'none'; 
    }, true);
}

function showGlobalDropdown(e, link, tweetElement, btnSource) {
    e.stopPropagation();
    
    // Calculate fixed position based on the button clicked
    const rect = e.target.getBoundingClientRect();
    
    globalDropdown.innerHTML = ''; // Clear previous options

    // Option 1: Add to Queue
    const optIngest = document.createElement('div');
    optIngest.className = 'vchat-dropdown-item';
    optIngest.innerHTML = '📥 Add to Queue';
    optIngest.onclick = (ev) => { 
        ev.stopPropagation(); 
        globalDropdown.style.display = 'none'; 
        handleIngest(link, btnSource); 
    };

    // Option 2: Queue + Comments
    const optIngestComm = document.createElement('div');
    optIngestComm.className = 'vchat-dropdown-item';
    optIngestComm.innerHTML = '💬 Queue + Comments';
    optIngestComm.onclick = (ev) => { 
        ev.stopPropagation(); 
        globalDropdown.style.display = 'none'; 
        handleIngestWithComments(link, tweetElement, btnSource); 
    };

    // Option 3: Scrape Comments
    const optScrapeComm = document.createElement('div');
    optScrapeComm.className = 'vchat-dropdown-item';
    optScrapeComm.innerHTML = '📝 Scrape Only Comments';
    optScrapeComm.onclick = (ev) => { 
        ev.stopPropagation(); 
        globalDropdown.style.display = 'none'; 
        handleScrapeComments(link, tweetElement, btnSource); 
    };

    globalDropdown.appendChild(optIngest);
    globalDropdown.appendChild(optIngestComm);
    globalDropdown.appendChild(optScrapeComm);

    // Position perfectly below the button
    globalDropdown.style.top = `${rect.bottom + 4}px`;
    globalDropdown.style.left = `${rect.left}px`;
    globalDropdown.style.display = 'block';
}


// ============================================================================
// MODULE: User Profile Scraper
// ============================================================================
class UserProfileScraper {
    constructor() {
        this.isScraping = false;
        this.scrapedPosts = new Map();
        this.observer = null;
        this.scrollInterval = null;
        this.targetHandle = null;
    }

    init() {
        const path = window.location.pathname;
        if (window.location.hostname.includes('x.com') || window.location.hostname.includes('twitter.com')) {
            const match = path.match(/^\/([^/]+)(?:\/.*)?$/);
            const reservedRoutes =['home', 'explore', 'notifications', 'messages', 'i', 'compose', 'settings', 'search', 'hashtag'];

            if (match && match[1] && !reservedRoutes.includes(match[1])) {
                this.targetHandle = match[1];
                this.injectScrapeControl(this.targetHandle);
            }
        }
    }

    injectScrapeControl(handle) {
        if (document.getElementById('vchat-profile-control')) return;

        const container = document.createElement('div');
        container.id = 'vchat-profile-control';
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 80px; 
            width: 300px;
            padding: 12px; 
            background: #292a2d; 
            border: 1px solid #3c4043; 
            border-radius: 4px; 
            color: #e8eaed; 
            font-family: "Google Sans", Roboto, sans-serif;
            font-size: 13px;
            z-index: 2147483647;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        `;
        
        container.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #3c4043; padding-bottom:8px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-size:16px;">🕵️</span>
                    <strong>@${handle}</strong>
                </div>
                <span id="vchat-scrape-status" style="font-family:monospace; color:#8ab4f8; font-size:11px;">Ready</span>
            </div>
            <div style="display:flex; gap:8px;">
                <button id="btn-start-scrape" style="background:#4285F4; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; font-weight:500; font-size:12px; flex:1;">Start Scrape</button>
                <button id="btn-stop-scrape" style="background:#EA4335; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; font-weight:500; font-size:12px; flex:1; display:none;">Stop</button>
            </div>
            <div id="vchat-scrape-log" style="margin-top:8px; height:60px; overflow-y:auto; font-family:'Roboto Mono', monospace; font-size:10px; color:#9aa0a6; background:#202124; padding:6px; border-radius:4px; border:1px solid #3c4043;"></div>
        `;

        document.body.appendChild(container);

        document.getElementById('btn-start-scrape').onclick = () => this.startScraping();
        document.getElementById('btn-stop-scrape').onclick = () => this.stopScraping();
    }

    log(msg) {
        const logEl = document.getElementById('vchat-scrape-log');
        if (logEl) {
            const line = document.createElement('div');
            line.innerText = `> ${msg}`;
            logEl.prepend(line);
        }
        console.log(`[vChat Scraper] ${msg}`);
        if (panelContent && !document.getElementById('vchat-analysis-result')) {
            const statusEl = document.getElementById('vchat-panel-status');
            if(statusEl) statusEl.innerText = msg;
        }
    }

    async startScraping() {
        this.isScraping = true;
        this.scrapedPosts.clear();
        document.getElementById('btn-start-scrape').style.display = 'none';
        document.getElementById('btn-stop-scrape').style.display = 'block';
        document.getElementById('vchat-scrape-status').innerText = "Scraping...";
        
        this.log("Starting scrape sequence...");
        let noNewPostsCount = 0;

        this.scrollInterval = setInterval(async () => {
            if (!this.isScraping) return;

            const countBefore = this.scrapedPosts.size;
            this.extractVisibleTweets();
            const countAfter = this.scrapedPosts.size;
            
            const delta = countAfter - countBefore;
            if (delta > 0) {
                this.log(`Found ${delta} new posts (Total: ${countAfter})`);
                noNewPostsCount = 0;
            } else {
                noNewPostsCount++;
                this.log("Scanning...");
            }

            window.scrollBy(0, 1500);

            if (noNewPostsCount > 10 || countAfter >= 150) {
                this.log("Stopping criteria met.");
                this.stopScraping();
            }
        }, 3000); 
    }

    extractVisibleTweets() {
        const tweets = document.querySelectorAll('article[data-testid="tweet"]');
        tweets.forEach(tweet => {
            try {
                const timeEl = tweet.querySelector('time');
                const textEl = tweet.querySelector('[data-testid="tweetText"]');
                const linkEl = tweet.querySelector('a[href*="/status/"]');
                
                if (linkEl && timeEl) {
                    const link = linkEl.href;
                    if (!this.scrapedPosts.has(link)) {
                        const postData = {
                            link: link,
                            timestamp: timeEl.getAttribute('datetime'),
                            text: textEl ? textEl.innerText : "[Media Only]",
                            is_reply: tweet.innerText.includes('Replying to'),
                            metrics: {
                                replies: this.parseMetric(tweet.querySelector('[data-testid="reply"]')),
                                reposts: this.parseMetric(tweet.querySelector('[data-testid="retweet"]')),
                                likes: this.parseMetric(tweet.querySelector('[data-testid="like"]')),
                                views: this.parseMetric(tweet.querySelector('a[href*="/analytics"]')) 
                            }
                        };
                        this.scrapedPosts.set(link, postData);
                    }
                }
            } catch (e) { }
        });
    }

    parseMetric(el) {
        if (!el) return 0;
        const txt = el.getAttribute('aria-label') || el.innerText || "";
        const numMatch = txt.match(/([\d\.]+)([KMB])?/);
        if (!numMatch) return 0;
        let val = parseFloat(numMatch[1]);
        const unit = numMatch[2];
        if (unit === 'K') val *= 1000;
        if (unit === 'M') val *= 1000000;
        if (unit === 'B') val *= 1000000000;
        return Math.floor(val);
    }

    stopScraping() {
        this.isScraping = false;
        clearInterval(this.scrollInterval);
        document.getElementById('btn-start-scrape').style.display = 'block';
        document.getElementById('btn-stop-scrape').style.display = 'none';
        document.getElementById('vchat-scrape-status').innerText = `Done (${this.scrapedPosts.size})`;
        
        this.log("Uploading data...");
        this.sendPayload();
    }

    async sendPayload() {
        const payload = {
            username: this.targetHandle,
            scraped_at: new Date().toISOString(),
            posts: Array.from(this.scrapedPosts.values())
        };
        
        chrome.runtime.sendMessage({ type: 'INGEST_USER_HISTORY', payload: payload }, (res) => {
            if (res && res.success) {
                this.log(`Uploaded ${res.new_posts} new posts.`);
                alert(`Successfully archived ${res.new_posts} posts for @${this.targetHandle}.`);
            } else {
                this.log("Upload failed: " + (res?.error || "Unknown"));
            }
        });
    }
}

// ============================================================================
// CORE EXTENSION LOGIC
// ============================================================================

function initSidePanel() {
    if (document.getElementById('vchat-panel')) return;

    sidePanel = document.createElement('div');
    sidePanel.id = 'vchat-panel';
    sidePanel.className = 'hidden'; 
    
    sidePanel.innerHTML = `
        <div class="vchat-panel-header">
            <span class="vchat-title">vChat <span>Agent</span></span>
            <button class="vchat-close-btn" id="vchat-close" title="Close">×</button>
        </div>
        <div class="vchat-tabs">
            <button class="vchat-tab active" data-tab="analysis">Analysis</button>
            <button class="vchat-tab" data-tab="labeling">Manual</button>
        </div>
        <div class="vchat-panel-content" id="vchat-content">
            <div class="vchat-status-msg" id="vchat-panel-status" style="color: #9aa0a6; text-align: center; margin-top: 40px;">
                Select a post and run ⚡ Analyze to view veracity factors here.
            </div>
        </div>
    `;
    
    document.body.appendChild(sidePanel);

    toggleBtn = document.createElement('div');
    toggleBtn.id = 'vchat-toggle';
    toggleBtn.innerHTML = '⚡';
    toggleBtn.title = "Open vChat Panel";
    toggleBtn.onclick = togglePanel;
    document.body.appendChild(toggleBtn);

    document.getElementById('vchat-close').onclick = togglePanel;
    panelContent = document.getElementById('vchat-content');

    sidePanel.querySelectorAll('.vchat-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            sidePanel.querySelectorAll('.vchat-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tabName = btn.getAttribute('data-tab');
            if (currentLink) renderPanelForLink(currentLink, currentCaption, currentPlatform, tabName);
        });
    });
}

function togglePanel() {
    if (sidePanel.classList.contains('hidden')) {
        sidePanel.classList.remove('hidden');
        toggleBtn.style.display = 'none';
    } else {
        sidePanel.classList.add('hidden');
        toggleBtn.style.display = 'flex';
    }
}

function injectTwitterButtons() {
    const tweets = document.querySelectorAll('article[data-testid="tweet"]');
    tweets.forEach(tweet => {
        const actionBar = tweet.querySelector('[role="group"]');
        if (actionBar && !actionBar.querySelector('.vchat-btn-group')) {
            const timeElement = tweet.querySelector('a[href*="/status/"]');
            const tweetLink = timeElement ? timeElement.href : window.location.href;
            const textElement = tweet.querySelector('[data-testid="tweetText"]');
            const caption = textElement ? textElement.innerText : "";

            createButtonUI(actionBar, tweetLink, caption, 'twitter', {}, tweet);
        }
    });
}

function createButtonUI(container, link, caption, platform, stats, tweetElement) {
    if (container.querySelector('.vchat-btn-group')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'vchat-btn-group';
    wrapper.onclick = (e) => e.stopPropagation();
    
    // Primary: Live Analyze
    const btnAnalyze = document.createElement('button');
    btnAnalyze.className = 'vchat-btn primary';
    btnAnalyze.innerHTML = '⚡ Analyze';
    btnAnalyze.title = "Live Factuality Analysis";
    btnAnalyze.onclick = (e) => { 
        e.stopPropagation(); 
        handleLiveAnalyze(link, caption, btnAnalyze, tweetElement); 
    };

    // Secondary: More Menu triggers Global Dropdown
    const btnMore = document.createElement('button');
    btnMore.className = 'vchat-btn secondary';
    btnMore.innerHTML = '⋮';
    btnMore.title = "More Actions";
    btnMore.onclick = (e) => showGlobalDropdown(e, link, tweetElement, btnMore);

    wrapper.appendChild(btnAnalyze);
    wrapper.appendChild(btnMore);
    container.appendChild(wrapper);
}

// ----------------------------------------------------------------------------
// HANDLERS
// ----------------------------------------------------------------------------

async function handleLiveAnalyze(link, caption, btn, tweetElement) {
    currentLink = link;
    currentCaption = caption;
    
    // Update UI State
    btn.innerHTML = '⏳ Analyzing...';
    btn.style.opacity = '0.7';
    if (sidePanel.classList.contains('hidden')) togglePanel();
    
    // Switch to Analysis Tab
    sidePanel.querySelectorAll('.vchat-tab').forEach(b => b.classList.remove('active'));
    sidePanel.querySelector('.vchat-tab[data-tab="analysis"]').classList.add('active');
    
    showAnalysisLoading(link);

    let comments =[];
    if (window.location.href.includes('/status/') && tweetElement) {
         try {
             comments = await scrapeLocalComments(tweetElement, 10);
         } catch(e) {
             console.error("Failed to scrape comments for live analyze", e);
         }
    }

    chrome.runtime.sendMessage({type: 'LIVE_ANALYZE', link: link, comments: comments}, (res) => {
        btn.style.opacity = '1';
        if (res && res.success && res.data && res.data.result && !res.data.error) {
            btn.innerHTML = '✔ Verified';
            btn.style.backgroundColor = '#34A853'; // Google Green
            btn.style.borderColor = '#34A853';
            renderAnalysisResult(res.data.result, link);
        } else {
            btn.innerHTML = '❌ Failed';
            btn.style.backgroundColor = '#EA4335'; // Google Red
            btn.style.borderColor = '#EA4335';
            
            const errObj = res?.data?.error || res?.error || "Unknown Server Error";
            const errStr = typeof errObj === 'object' ? (errObj.message || JSON.stringify(errObj)) : errObj;
            renderAnalysisError(errStr, link);
        }
        
        setTimeout(() => { 
            btn.innerHTML = '⚡ Analyze'; 
            btn.style.backgroundColor = ''; 
            btn.style.borderColor = ''; 
        }, 4000);
    });
}

function showAnalysisLoading(link) {
    panelContent.innerHTML = `
        <div class="vchat-section-title">A2A Pipeline Active</div>
        <div class="vchat-link-preview">${link}</div>
        <div class="vchat-loading-pulse">
            <div>Agent orchestrating models...</div>
            <div style="font-weight:normal; color:#9aa0a6; margin-top:8px;">Extracting visual/audio factors...</div>
        </div>
    `;
}

function renderAnalysisResult(result, link) {
    const data = result.data; // This is the 'parsed_data' or final_result from agent_logic
    if (!data || !data.final_assessment) {
        renderAnalysisError("Agent returned empty or unparsed data.", link);
        return;
    }

    const score = parseInt(data.final_assessment.veracity_score_total || 0);
    let scoreColorClass = 'med';
    if (score >= 75) scoreColorClass = 'high';
    if (score <= 40) scoreColorClass = 'low';

    const v = data.veracity_vectors || {};
    const m = data.modalities || {};

    panelContent.innerHTML = `
        <div class="vchat-section-title">Agentic Assessment</div>
        <div class="vchat-link-preview">${link}</div>
        
        <div class="vchat-score-card">
            <div class="label">Final Veracity Score</div>
            <div class="score ${scoreColorClass}">${score}<span style="font-size:18px; color:#9aa0a6;">/100</span></div>
            <div style="font-size:11px; color:#9aa0a6; margin-top:4px; text-transform:uppercase;">
                ${data.disinformation_analysis?.classification || 'Unknown'}
            </div>
        </div>

        <div class="vchat-reasoning-box">
            <b>Reasoning:</b> ${data.final_assessment.reasoning}
        </div>

        <div style="margin-bottom: 16px; background: #292a2d; padding: 12px; border-radius: 6px; border: 1px solid #3c4043;">
            <div class="vchat-section-title" style="margin-bottom:8px; border-bottom:none;">Veracity Vectors</div>
            <div class="vchat-vector-row"><span class="v-name">Visual Integrity</span><span class="v-score">${v.visual_integrity_score || 'N/A'}/10</span></div>
            <div class="vchat-vector-row"><span class="v-name">Audio Integrity</span><span class="v-score">${v.audio_integrity_score || 'N/A'}/10</span></div>
            <div class="vchat-vector-row"><span class="v-name">Logical Consistency</span><span class="v-score">${v.logical_consistency_score || 'N/A'}/10</span></div>
            <div class="vchat-vector-row"><span class="v-name">Source Credibility</span><span class="v-score">${v.source_credibility_score || 'N/A'}/10</span></div>
            
            <div class="vchat-section-title" style="margin: 12px 0 8px; border-bottom:none;">Alignments</div>
            <div class="vchat-vector-row"><span class="v-name">Video-Caption</span><span class="v-score">${m.video_caption_score || 'N/A'}/10</span></div>
            <div class="vchat-vector-row"><span class="v-name">Audio-Caption</span><span class="v-score">${m.audio_caption_score || 'N/A'}/10</span></div>
        </div>
        
        <div style="font-size:10px; color:#9aa0a6; text-align:center;">Data synced to local Manager.</div>
    `;
}

function renderAnalysisError(errorMsg, link) {
    panelContent.innerHTML = `
        <div class="vchat-section-title" style="color:#EA4335;">Pipeline Error</div>
        <div class="vchat-link-preview">${link}</div>
        <div style="background:rgba(234, 67, 53, 0.1); color:#f28b82; padding:12px; border-radius:4px; font-size:12px; border:1px solid rgba(234, 67, 53, 0.3); word-break: break-word;">
            ${errorMsg}
        </div>
    `;
}

function renderPanelForLink(link, caption, platform, tab) {
    if (tab === 'analysis') {
        if (!document.getElementById('vchat-analysis-result') && !panelContent.innerHTML.includes('vchat-score-card')) {
             panelContent.innerHTML = `<div class="vchat-status-msg" style="color: #9aa0a6; text-align: center; margin-top: 40px;">Select a post and run ⚡ Analyze to view results here.</div>`;
        }
    } else if (tab === 'labeling') {
        panelContent.innerHTML = `
            <div class="vchat-section-title">Manual Override</div>
            <p style="color:#9aa0a6; font-size:11px; margin-bottom:12px;">Reviewing: ${link}</p>
            <button id="btn-open-full" class="vchat-action-btn">Open Editor Form</button>
        `;
        document.getElementById('btn-open-full').onclick = () => {
             window.open(chrome.runtime.getURL(`src/manual_label.html?link=${encodeURIComponent(link)}&caption=${encodeURIComponent(caption)}`), '_blank', 'width=600,height=800');
        };
    }
}

// Scrapes ~10 comments and returns them, including their links
async function scrapeLocalComments(tweetElement, amount=10) {
    if (!window.location.href.includes('/status/')) return[];
    window.scrollBy(0, 500);
    await new Promise(r => setTimeout(r, 1000));
    const comments =[];
    document.querySelectorAll('article[data-testid="tweet"]').forEach(node => {
        if(node === tweetElement) return;
        const textNode = node.querySelector('[data-testid="tweetText"]');
        const userNode = node.querySelector('[data-testid="User-Name"]');
        
        // Extract comment permalink
        const timeEl = node.querySelector('time');
        let commentLink = "";
        if (timeEl) {
            const aTag = timeEl.closest('a');
            if (aTag && aTag.href) commentLink = aTag.href;
        } else {
            const aTag = node.querySelector('a[href*="/status/"]');
            if (aTag && aTag.href) commentLink = aTag.href;
        }

        if(textNode) comments.push({ 
            author: userNode ? userNode.innerText.split('\n')[0] : "Unknown", 
            text: textNode.innerText,
            link: commentLink
        });
    });
    return comments.slice(0, amount);
}

async function handleIngestWithComments(link, tweetElement, btnSource) {
    const originalText = btnSource.innerHTML;
    btnSource.innerHTML = '⏳ Queueing...';
    try {
        if (!window.location.href.includes('/status/')) { handleIngest(link, btnSource); return; }
        const comments = await scrapeLocalComments(tweetElement, 10);
        chrome.runtime.sendMessage({ type: 'INGEST_LINK_COMMENTS', link: link, comments: comments }, (res) => {
             if (res && res.success) btnSource.innerHTML = '✔ Added';
             else btnSource.innerHTML = '❌ Error';
             setTimeout(() => { btnSource.innerHTML = originalText; }, 2000);
        });
    } catch (e) { btnSource.innerHTML = '❌ Error'; }
}

async function handleScrapeComments(link, tweetElement, btnSource) {
    const originalText = btnSource.innerHTML;
    const amountStr = prompt("How many comments to sample?", "30");
    if (!amountStr) return;
    const amount = parseInt(amountStr) || 30;
    btnSource.innerHTML = '⏳ Scraping...';
    try {
        if (!window.location.href.includes('/status/')) {
            alert("Please open the post detail page first to scrape comments.");
            btnSource.innerHTML = originalText; return;
        }
        window.scrollBy(0, 500); await new Promise(r => setTimeout(r, 1000));
        window.scrollBy(0, 1000); await new Promise(r => setTimeout(r, 1500));
        const comments =[];
        document.querySelectorAll('article[data-testid="tweet"]').forEach(node => {
            if(node === tweetElement) return;
            const textNode = node.querySelector('[data-testid="tweetText"]');
            const userNode = node.querySelector('[data-testid="User-Name"]');
            
            // Extract comment permalink
            const timeEl = node.querySelector('time');
            let commentLink = "";
            if (timeEl) {
                const aTag = timeEl.closest('a');
                if (aTag && aTag.href) commentLink = aTag.href;
            } else {
                const aTag = node.querySelector('a[href*="/status/"]');
                if (aTag && aTag.href) commentLink = aTag.href;
            }

            if(textNode) comments.push({ 
                author: userNode ? userNode.innerText.split('\n')[0] : "Unknown", 
                text: textNode.innerText,
                link: commentLink
            });
        });
        
        chrome.runtime.sendMessage({ type: 'SAVE_COMMENTS', payload: { link: link, comments: comments.slice(0, amount) } }, (res) => {
            if(res && res.success) { btnSource.innerHTML = '✔ Saved'; alert(`Saved ${res.count} comments.`); } 
            else btnSource.innerHTML = '❌ Error';
            setTimeout(() => { btnSource.innerHTML = originalText; }, 3000);
        });
    } catch(e) { btnSource.innerHTML = '❌ Error'; }
}

function handleIngest(link, btnSource) {
    const originalText = btnSource.innerHTML;
    btnSource.innerHTML = '⏳ Adding...';
    chrome.runtime.sendMessage({type: 'INGEST_LINK', link: link}, (res) => {
        if (res && res.success) btnSource.innerHTML = '✔ Queued';
        else btnSource.innerHTML = '❌ Error';
        setTimeout(() => { btnSource.innerHTML = originalText; }, 2000);
    });
}

const scraper = new UserProfileScraper();
initSidePanel();
initGlobalDropdown();

const observer = new MutationObserver(() => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        scraper.init(); 
        injectTwitterButtons();
    }, 500); 
});

observer.observe(document.body, { childList: true, subtree: true });
