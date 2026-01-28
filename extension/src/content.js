// content.js - Injects buttons, Side Panel, and Handles Scraping
console.log("[vChat] Extension loaded.");

let debounceTimer = null;
let sidePanel = null;
let panelContent = null;
let toggleBtn = null;

// State
let currentLink = "";
let currentCaption = "";
let currentPlatform = "";
let currentStats = {};

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
            const reservedRoutes = ['home', 'explore', 'notifications', 'messages', 'i', 'compose', 'settings', 'search', 'hashtag'];

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
        // Fixed position bottom-right ensures it stays visible during scrolling
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 80px; 
            width: 300px;
            padding: 12px; 
            background: #0f172a; 
            border: 1px solid #6366f1; 
            border-radius: 8px; 
            color: #e2e8f0; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            z-index: 2147483647;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        `;
        
        container.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #334155; padding-bottom:8px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-size:16px;">🕵️</span>
                    <strong>@${handle}</strong>
                </div>
                <span id="vchat-scrape-status" style="font-family:monospace; color:#818cf8; font-size:11px;">Ready</span>
            </div>
            <div style="display:flex; gap:8px;">
                <button id="btn-start-scrape" style="background:#4f46e5; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; font-weight:600; font-size:12px; flex:1;">Start Scrape</button>
                <button id="btn-stop-scrape" style="background:#be123c; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; font-weight:600; font-size:12px; flex:1; display:none;">Stop</button>
            </div>
            <div id="vchat-scrape-log" style="margin-top:8px; height:60px; overflow-y:auto; font-family:monospace; font-size:10px; color:#94a3b8; background:#1e293b; padding:4px; border-radius:4px;"></div>
        `;

        document.body.appendChild(container);

        document.getElementById('btn-start-scrape').onclick = () => this.startScraping();
        document.getElementById('btn-stop-scrape').onclick = () => this.stopScraping();
        
        this.log(`Panel injected for @${handle}`);
    }

    log(msg) {
        const logEl = document.getElementById('vchat-scrape-log');
        if (logEl) {
            const line = document.createElement('div');
            line.innerText = `> ${msg}`;
            logEl.prepend(line);
        }
        console.log(`[vChat Scraper] ${msg}`);
        
        // Also update the main side panel if it's open
        if (panelContent) {
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
        
        // Add Stop button to main panel as well for accessibility
        if (panelContent) {
             const stopBtn = document.createElement('button');
             stopBtn.innerText = "STOP SCRAPE";
             stopBtn.className = "vchat-action-btn";
             stopBtn.style.background = "#be123c";
             stopBtn.onclick = () => this.stopScraping();
             stopBtn.id = "panel-stop-btn";
             
             // Prepend to panel content
             panelContent.insertBefore(stopBtn, panelContent.firstChild);
        }

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

            // Scroll down
            window.scrollBy(0, 1500);

            // Auto-stop conditions
            if (noNewPostsCount > 10) {
                this.log("No new content found. Stopping.");
                this.stopScraping();
            }
            if (countAfter >= 150) { 
                this.log("Batch limit reached. Stopping.");
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
        document.getElementById('vchat-scrape-status').innerText = `Done (${this.scrapedPosts.size} posts)`;
        
        // Remove stop button from main panel
        const pBtn = document.getElementById("panel-stop-btn");
        if(pBtn) pBtn.remove();
        
        this.log("Scraping finished. Uploading data...");
        this.sendPayload();
    }

    async sendPayload() {
        const payload = {
            username: this.targetHandle,
            scraped_at: new Date().toISOString(),
            posts: Array.from(this.scrapedPosts.values())
        };
        
        chrome.runtime.sendMessage({
            type: 'INGEST_USER_HISTORY',
            payload: payload
        }, (res) => {
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
            <span class="vchat-title">vChat Assistant</span>
            <button class="vchat-close-btn" id="vchat-close" title="Close">×</button>
        </div>
        <div class="vchat-tabs">
            <button class="vchat-tab active" data-tab="comments">💬 Context</button>
            <button class="vchat-tab" data-tab="labeling">📝 Labeling</button>
        </div>
        <div class="vchat-panel-content" id="vchat-content">
            <div class="vchat-status-msg" id="vchat-panel-status">
                Select a post to begin.
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
            if (currentLink) {
                renderPanelForLink(currentLink, currentCaption, currentPlatform, tabName);
            }
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
             const stats = {
                likes: 0,
                shares: 0,
                comments: 0,
                platform: 'twitter'
            };

            createButtonUI(actionBar, tweetLink, caption, 'twitter', stats, tweet);
        }
    });
}

function createButtonUI(container, link, caption, platform, stats, tweetElement) {
    if (container.querySelector('.vchat-btn-group')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'vchat-btn-group';
    wrapper.style.display = 'flex';
    wrapper.style.gap = '6px';
    wrapper.style.marginLeft = '10px';
    wrapper.style.zIndex = '999';
    wrapper.style.position = 'relative';

    wrapper.onclick = (e) => e.stopPropagation();
    
    // Ingest
    const btnIngest = document.createElement('button');
    btnIngest.className = 'vchat-btn';
    btnIngest.innerHTML = '⚡';
    btnIngest.title = "Add to Queue";
    btnIngest.onclick = (e) => { e.stopPropagation(); handleIngest(link, btnIngest); };
    
    // Analyze
    const btnOpen = document.createElement('button');
    btnOpen.className = 'vchat-btn comments';
    btnOpen.innerHTML = '🔍';
    btnOpen.title = "Analyze Veracity";
    btnOpen.onclick = (e) => { 
        e.stopPropagation(); 
        openPanel(link, caption, platform, stats); 
    };

    // Scrape Comments
    const btnComments = document.createElement('button');
    btnComments.className = 'vchat-btn';
    btnComments.style.backgroundColor = '#10b981';
    btnComments.innerHTML = '💬';
    btnComments.title = "Scrape Comments";
    btnComments.onclick = (e) => {
        e.stopPropagation();
        handleScrapeComments(link, tweetElement, btnComments);
    };

    wrapper.appendChild(btnIngest);
    wrapper.appendChild(btnOpen);
    wrapper.appendChild(btnComments);
    container.appendChild(wrapper);
}

async function handleScrapeComments(link, tweetElement, btn) {
    // Prompt for amount
    const amountStr = prompt("How many comments to sample?", "30");
    if (!amountStr) return;
    const amount = parseInt(amountStr) || 30;

    btn.innerHTML = '⏳';
    
    try {
        if (!window.location.href.includes('/status/')) {
            alert("Please open the post detail page first to scrape comments.");
            btn.innerHTML = '💬';
            return;
        }

        // Auto-scroll to load comments
        window.scrollBy(0, 500);
        await new Promise(r => setTimeout(r, 1000));
        window.scrollBy(0, 1000);
        await new Promise(r => setTimeout(r, 1500));

        const comments = [];
        const replyNodes = document.querySelectorAll('article[data-testid="tweet"]');
        
        replyNodes.forEach(node => {
            if(node === tweetElement) return;
            
            const textNode = node.querySelector('[data-testid="tweetText"]');
            const userNode = node.querySelector('[data-testid="User-Name"]');
            
            if(textNode) {
                comments.push({
                    author: userNode ? userNode.innerText.split('\n')[0] : "Unknown",
                    text: textNode.innerText
                });
            }
        });

        // Limit to requested amount
        const sample = comments.slice(0, amount);
        
        chrome.runtime.sendMessage({
            type: 'SAVE_COMMENTS',
            payload: {
                link: link,
                comments: sample
            }
        }, (res) => {
            if(res && res.success) {
                btn.innerHTML = '✔';
                alert(`Saved ${res.count} comments to data/comments/`);
            } else {
                btn.innerHTML = '❌';
            }
            setTimeout(() => { btn.innerHTML = '💬'; }, 3000);
        });

    } catch(e) {
        console.error(e);
        btn.innerHTML = '❌';
    }
}

function handleIngest(link, btn) {
    btn.innerHTML = '...';
    chrome.runtime.sendMessage({type: 'INGEST_LINK', link: link}, (res) => {
        if (res && res.success) {
            btn.innerHTML = '✔';
            btn.style.backgroundColor = '#10b981';
        } else {
            btn.innerHTML = '❌';
            btn.style.backgroundColor = '#ef4444';
        }
        setTimeout(() => { btn.innerHTML = '⚡'; btn.style.backgroundColor = '#6366f1'; }, 2000);
    });
}

function openPanel(link, caption, platform, stats) {
    currentLink = link;
    currentCaption = caption;
    currentPlatform = platform;
    currentStats = stats;
    if (sidePanel.classList.contains('hidden')) togglePanel();
    const activeTab = sidePanel.querySelector('.vchat-tab.active').getAttribute('data-tab');
    renderPanelForLink(link, caption, platform, activeTab);
}

function renderPanelForLink(link, caption, platform, tab) {
    if (tab === 'comments') {
        panelContent.innerHTML = `<div class="vchat-section-title">Analysis</div><div class="vchat-link-preview">${link}</div><div>Use the manual tab to label.</div>`;
    } else {
        renderLabelingTab(link, caption);
    }
}

function renderLabelingTab(link, caption) {
    panelContent.innerHTML = `
        <div class="vchat-section-title">Manual Labeling</div>
        <p style="color:#94a3b8; font-size:11px;">Labeling: ${link}</p>
        <button id="btn-open-full" class="vchat-action-btn">Open Full Form</button>
    `;
    document.getElementById('btn-open-full').onclick = () => {
         window.open(chrome.runtime.getURL(`src/manual_label.html?link=${encodeURIComponent(link)}&caption=${encodeURIComponent(caption)}`), '_blank', 'width=600,height=800');
    };
}


const scraper = new UserProfileScraper();
initSidePanel();

const observer = new MutationObserver(() => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        scraper.init(); 
        injectTwitterButtons();
    }, 500); 
});

observer.observe(document.body, { childList: true, subtree: true });
