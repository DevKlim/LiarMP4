import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, Play, Upload, List, Database, BarChart2, ExternalLink, 
  Users, MessageSquare, TrendingUp, ShieldCheck, UserCheck, Search, PlusCircle, 
  StopCircle, RefreshCw, CheckCircle2, PenTool, ClipboardCheck, Info, Clock, FileText,
  Tag, Home, Cpu, FlaskConical, Target, Trash2, ArrowUpRight, CheckSquare, Square,
  Layers, Activity, Zap, BrainCircuit, Network, Archive
} from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [logs, setLogs] = useState<string>('System Ready.\n');
  const [isProcessing, setIsProcessing] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);
  
  // Processing Config State
  const [modelProvider, setModelProvider] = useState('vertex');
  const [apiKey, setApiKey] = useState('');
  const [modelName, setModelName] = useState('gemini-1.5-pro-preview-0409');
  const [projectId, setProjectId] = useState('');
  const [location, setLocation] = useState('us-central1');
  const [includeComments, setIncludeComments] = useState(false);
  const [reasoningMethod, setReasoningMethod] = useState('cot');
  const [promptTemplate, setPromptTemplate] = useState('standard');
  const [availablePrompts, setAvailablePrompts] = useState<any[]>([]);

  // Predictive Config
  const [predictiveModelType, setPredictiveModelType] = useState('logistic');
  const [predictiveResult, setPredictiveResult] = useState<any>(null);

  // Data States
  const [queueList, setQueueList] = useState<any[]>([]);
  const [profileList, setProfileList] = useState<any[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<any>(null);
  const [profilePosts, setProfilePosts] = useState<any[]>([]);
  const [communityDatasets, setCommunityDatasets] = useState<any[]>([]);
  const [communityAnalysis, setCommunityAnalysis] = useState<any>(null);
  const [integrityBoard, setIntegrityBoard] = useState<any[]>([]);
  const [datasetList, setDatasetList] = useState<any[]>([]);
  const [benchmarks, setBenchmarks] = useState<any>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Selection State
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());

  // Manual Labeling State
  const [manualLink, setManualLink] = useState('');
  const [manualCaption, setManualCaption] = useState('');
  const [manualTags, setManualTags] = useState('');
  const [manualReasoning, setManualReasoning] = useState('');
  const [manualScores, setManualScores] = useState({
      visual: 5, audio: 5, source: 5, logic: 5, emotion: 5,
      va: 5, vc: 5, ac: 5, final: 50
  });
  
  const [labelBrowserMode, setLabelBrowserMode] = useState<'queue' | 'dataset'>('queue');
  const [labelFilter, setLabelFilter] = useState('');

  // Placeholder Benchmark Data for Home Screen
  const modelLeaderboard = {
    predictive: [
      { name: "XGBoost", desc: "Visual Integrity", score: 64.2, color: "bg-sky-500", text: "text-sky-400" },
      { name: "CatBoost (Hybrid)", desc: "Visual Integrity + Audio Integrity + Visual-Audio Semantic", score: 65.3, color: "bg-sky-400", text: "text-sky-300" },
    ],
    generative: [
      { name: "Gemini 2.5 Flash", desc: "Standard CoT & Prompt, Visual Integrity", score: 65.5, color: "bg-indigo-500", text: "text-indigo-400" },
      { name: "Gemini 2.5 Flash Lite", desc: "Fractal CoT + Standard Prompt", score: 45.4, color: "bg-purple-500", text: "text-purple-400" },
            { name: "Gemini 2.5 Flash", desc: "Standard CoT + Alt Prompt", score: 63.5, color: "bg-indigo-500", text: "text-indigo-400" },
      { name: "Gemini 2.5 Flash Lite", desc: "Fractal CoT + Alt Prompt", score: 46.4, color: "bg-purple-500", text: "text-purple-400" },
    ],
    agentic: [
      { name: "Single Agent + Tools", desc: "Google Search", score: 53, color: "bg-emerald-500", text: "text-emerald-400" },
      { name: "Multi-Agent", desc: "Researcher + Critic + Judge", score: 73.4, color: "bg-orange-500", text: "text-orange-400" },
    ]
  };

  // --- Handlers Defined Early to Avoid ReferenceError ---
  const handleTagsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setManualTags(e.target.value);
  };

  useEffect(() => {
    const load = async (url: string, setter: any) => {
        try { const res = await fetch(url); const d = await res.json(); setter(Array.isArray(d) ? d : (d.status==='no_data'?null:d)); } catch(e) {}
    };

    load('/config/prompts', setAvailablePrompts);
    if (activeTab === 'home') load('/benchmarks/stats', setBenchmarks);
    if (activeTab === 'queue') load('/queue/list', setQueueList);
    if (activeTab === 'profiles') load('/profiles/list', setProfileList);
    if (activeTab === 'community') load('/community/list_datasets', setCommunityDatasets);
    if (activeTab === 'analytics') load('/analytics/account_integrity', setIntegrityBoard);
    if (activeTab === 'dataset' || activeTab === 'manual' || activeTab === 'groundtruth') load('/dataset/list', setDatasetList);
    if (activeTab === 'manual') load('/queue/list', setQueueList);
    
    setSelectedItems(new Set());
  }, [activeTab, refreshTrigger]);

  useEffect(() => {
    if (logContainerRef.current) logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
  }, [logs]);

  // Helpers: Robust Tag Extraction
  const existingTags = React.useMemo(() => {
    const tags = new Set<string>();
    if (datasetList && Array.isArray(datasetList)) {
        datasetList.forEach(item => {
            if (item.tags && typeof item.tags === 'string') {
                // Robust Split: Handles commas, newlines, pipes, or semicolons
                item.tags.split(/[,\n|;]+/).forEach((t: string) => {
                    const clean = t.trim().replace(/^['"]|['"]$/g, ''); // Remove quotes
                    if (clean.length > 1) tags.add(clean);
                });
            }
        });
    }
    return Array.from(tags).sort();
  }, [datasetList]);

  const toggleTag = (tag: string) => {
    let current = manualTags.split(',').map(t => t.trim()).filter(Boolean);
    if (current.includes(tag)) {
        current = current.filter(t => t !== tag);
    } else {
        current.push(tag);
    }
    setManualTags(current.join(', '));
  };

  const getTweetId = (link: string) => {
    const match = link.match(/(?:twitter|x)\.com\/[^/]+\/status\/(\d+)/);
    return match ? match[1] : null;
  };

  const loadProfilePosts = async (username: string) => {
      const res = await fetch(`/profiles/${username}/posts`);
      const data = await res.json();
      setProfilePosts(data);
      setSelectedProfile(username);
  };

  const sendToManualLabeler = (link: string, text: string) => {
      setManualLink(link);
      setManualCaption(text);
      setManualScores({
          visual: 5, audio: 5, source: 5, logic: 5, emotion: 5,
          va: 5, vc: 5, ac: 5, final: 50
      });
      setManualReasoning('');
      setManualTags('');
      setActiveTab('manual');
  };
  
  const loadFromBrowser = (item: any, mode: 'queue' | 'dataset') => {
      setManualLink(item.link);
      if (mode === 'dataset') {
          setManualCaption(item.caption || '');
          setManualTags(item.tags || '');
          setManualReasoning(item.reasoning || item.final_reasoning || '');
          setManualScores({
              visual: parseInt(item.visual_integrity_score || item.visual_score) || 5,
              audio: parseInt(item.audio_integrity_score || item.audio_score) || 5,
              source: parseInt(item.source_credibility_score) || 5, 
              logic: parseInt(item.logical_consistency_score || item.logic_score) || 5,
              emotion: parseInt(item.emotional_manipulation_score) || 5,
              va: parseInt(item.video_audio_score) || 5, 
              vc: parseInt(item.video_caption_score || item.align_video_caption) || 5,
              ac: parseInt(item.audio_caption_score) || 5,
              final: parseInt(item.final_veracity_score) || 50
          });
      } else {
          setManualCaption(''); setManualReasoning(''); setManualTags('');
          setManualScores({
              visual: 5, audio: 5, source: 5, logic: 5, emotion: 5,
              va: 5, vc: 5, ac: 5, final: 50
          });
      }
  };

  const toggleSelection = (id: string) => {
      const newSet = new Set(selectedItems);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      setSelectedItems(newSet);
  };

  const promoteSelected = async () => {
      if (selectedItems.size === 0) return alert("No items selected.");
      if (!confirm(`Promote ${selectedItems.size} items to Ground Truth?`)) return;
      
      try {
          const res = await fetch('/manual/promote', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ ids: Array.from(selectedItems) })
          });
          
          if (!res.ok) {
               const errText = await res.text();
               console.error("Promote Error:", errText);
               alert("Server Error during promotion: " + errText);
               return;
          }

          const d = await res.json();
          if(d.status === 'success') {
              alert(`Successfully promoted ${d.promoted_count} items.`);
              setSelectedItems(new Set());
              setRefreshTrigger(p => p+1);
          } else {
              alert("Promotion failed: " + (d.message || "Unknown error"));
          }
      } catch(e: any) { 
          console.error("Network Error:", e);
          alert("Network error: " + e.toString()); 
      }
  };

  const deleteSelected = async () => {
      if (selectedItems.size === 0) return alert("No items selected.");
      if (!confirm(`Delete ${selectedItems.size} items from Ground Truth? Irreversible.`)) return;

      try {
          const res = await fetch('/manual/delete', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ ids: Array.from(selectedItems) })
          });
          const d = await res.json();
          if(d.status === 'success') {
              alert(`Deleted ${d.deleted_count} items.`);
              setSelectedItems(new Set());
              setRefreshTrigger(p => p+1);
          } else alert("Error deleting: " + d.message);
      } catch(e) { alert("Network error."); }
  };

  const submitManualLabel = async () => {
      if(!manualLink) return alert("Link is required.");
      const payload = {
          link: manualLink, caption: manualCaption, tags: manualTags, reasoning: manualReasoning,
          visual_integrity_score: manualScores.visual, audio_integrity_score: manualScores.audio,
          source_credibility_score: manualScores.source, logical_consistency_score: manualScores.logic,
          emotional_manipulation_score: manualScores.emotion,
          video_audio_score: manualScores.va, video_caption_score: manualScores.vc, audio_caption_score: manualScores.ac,
          final_veracity_score: manualScores.final
      };
      
      try {
          const res = await fetch('/manual/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
          
          if(!res.ok) {
             const txt = await res.text();
             alert("Error saving: " + txt);
             return;
          }

          const d = await res.json();
          if(d.status === 'success') { 
              alert("Label Saved! Data added to Ground Truth."); 
              setRefreshTrigger(p => p+1); 
          } else {
              alert("Error saving label: " + (d.message || "Unknown Error"));
          }
      } catch(e: any) { 
          console.error(e);
          alert("Network error: " + e.toString()); 
      }
  };

  const analyzeComments = async (id: string) => {
      setCommunityAnalysis({ verdict: "Analyzing..." });
      const res = await fetch('/community/analyze', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ dataset_id: id })
      });
      setCommunityAnalysis(await res.json());
  };

  const runPredictiveTraining = async (useVisual: boolean) => {
      setPredictiveResult({ status: 'training' });
      try {
          const res = await fetch('/benchmarks/train_predictive', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ use_visual_meta: useVisual, model_type: predictiveModelType })
          });
          setPredictiveResult(await res.json());
      } catch (e) { setPredictiveResult({ error: "Failed to train." }); }
  };

  const queueUnlabeledPosts = async () => {
      const unlabeled = profilePosts.filter(p => !p.is_labeled).map(p => p.link);
      if(unlabeled.length === 0) return alert("All posts already labeled!");
      const csvContent = "link\n" + unlabeled.join("\n");
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const fd = new FormData(); fd.append("file", blob, "batch_upload.csv");
      try {
        await fetch('/queue/upload_csv', { method: 'POST', body: fd });
        alert(`Queued ${unlabeled.length} links.`); setRefreshTrigger(p => p+1);
      } catch (e) { alert("Error uploading."); }
  };

  const startProcessing = async () => {
      if (isProcessing) return;
      setIsProcessing(true);
      setLogs(prev => prev + '\n[SYSTEM] Starting Queue Processing...\n');
      const fd = new FormData();
      fd.append('model_selection', modelProvider); fd.append('gemini_api_key', apiKey);
      fd.append('gemini_model_name', modelName); fd.append('vertex_project_id', projectId);
      fd.append('vertex_location', location); fd.append('vertex_model_name', modelName);
      fd.append('include_comments', includeComments.toString()); fd.append('reasoning_method', reasoningMethod);
      fd.append('prompt_template', promptTemplate);

      try {
          const res = await fetch('/queue/run', { method: 'POST', body: fd });
          const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
          while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              if (value.includes('event: close')) { setIsProcessing(false); break; }
              const clean = value.replace(/data: /g, '').trim();
              if (clean) setLogs(prev => prev + clean + '\n');
          }
      } catch (e) { setIsProcessing(false); }
      setRefreshTrigger(p => p+1);
  };

  return (
    <div className="flex h-screen w-full bg-[#09090b] text-slate-200 font-sans overflow-hidden">
      
      {/* SIDEBAR */}
      <div className="w-[280px] flex flex-col border-r border-slate-800/60 bg-[#0c0c0e]">
        <div className="h-16 flex items-center px-6 border-b border-slate-800/60">
          <h1 className="text-sm font-bold text-white">vChat <span className="text-slate-500">Manager</span></h1>
        </div>
        <div className="flex-1 p-4 space-y-1">
           {[
               {id:'home', l:'Home & Benchmarks', i:Home},
               {id:'predictive', l:'Predictive Sandbox', i:FlaskConical},
               {id:'queue', l:'Ingest Queue', i:List}, 
               {id:'profiles', l:'User Profiles', i:Users}, 
               {id:'manual', l:'Labeling Studio', i:PenTool},
               {id:'dataset', l:'Data Manager', i:Archive}, // Renamed & Icon Changed
               {id:'groundtruth', l:'Ground Truth (Verified)', i:ShieldCheck},
               {id:'community', l:'Community Trust', i:MessageSquare}, 
               {id:'analytics', l:'Analytics', i:BarChart2}
           ].map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 text-xs font-medium rounded-lg ${activeTab===t.id ? 'bg-indigo-600/20 text-indigo-300' : 'text-slate-500 hover:bg-white/5'}`}>
                <t.i className="w-4 h-4" /> {t.l}
              </button>
           ))}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col bg-[#09090b] overflow-hidden">
         <div className="h-16 border-b border-slate-800/60 flex items-center px-8 bg-[#09090b]">
            <span className="text-sm font-bold text-white uppercase tracking-wider">{activeTab}</span>
         </div>

         <div className="flex-1 p-6 overflow-hidden flex flex-col">
            
            {/* HOME TAB */}
            {activeTab === 'home' && (
                <div className="h-full overflow-y-auto space-y-8 max-w-5xl pr-2">
                    <div className="grid grid-cols-3 gap-6">
                        <div className="col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                            <h2 className="text-xl font-bold text-white mb-4">Philosophy & Methodology</h2>
                            <p className="text-sm text-slate-400 mb-4">
                                The goal of this research is to test various predictive models, generative AI models, prompting techniques, and agents against a rigorous <strong>Ground Truth</strong> standard.
                            </p>
                            <p className="text-sm text-slate-400 mb-4">
                                By benchmarking "Standard" vs. "Fractal Chain-of-Thought (FCoT)" reasoning, we calculate accuracy deterministically. The platform enables Human-in-the-Loop calibration to verify "honesty" in AI outputs.
                            </p>
                            <div className="p-3 bg-indigo-900/20 border border-indigo-500/20 rounded-lg text-xs text-indigo-300 font-mono">
                                <strong>Procedure:</strong> Ingest Content → Run Generative Inference → Verify against Ground Truth → Calculate Delta.
                            </div>
                        </div>
                        <div className="bg-indigo-900/20 border border-indigo-500/30 rounded-xl p-6 flex flex-col justify-center items-center">
                            <div className="text-xs uppercase text-indigo-400 font-bold mb-2">Ground Truth Accuracy</div>
                            {benchmarks ? (
                                <>
                                    <div className="text-5xl font-mono font-bold text-white mb-2">{benchmarks.accuracy_percent}%</div>
                                    <div className="text-xs text-slate-500">MAE: {benchmarks.mae} points</div>
                                    <div className="text-xs text-slate-500 mt-2">{benchmarks.count} verified samples</div>
                                </>
                            ) : (
                                <span className="text-slate-600">No data found</span>
                            )}
                        </div>
                    </div>

                    {/* NEW: Model Architecture Leaderboard */}
                    <div>
                         <h3 className="text-sm font-bold text-slate-300 uppercase mb-4 flex items-center gap-2">
                            <Zap className="w-4 h-4"/> Architecture Leaderboard (Benchmark Placeholders)
                        </h3>
                        <div className="grid grid-cols-3 gap-6">
                            {/* Predictive */}
                            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                                <div className="flex items-center gap-2 text-sky-400 font-bold text-xs uppercase"><Activity className="w-4 h-4"/> Predictive Models</div>
                                {modelLeaderboard.predictive.map((m, i) => (
                                    <div key={i} className="bg-slate-950 p-3 rounded border border-slate-800">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-xs font-bold text-white">{m.name}</span>
                                            <span className={`text-xs font-mono font-bold ${m.text}`}>{m.score}%</span>
                                        </div>
                                        <div className="w-full bg-slate-900 h-1.5 rounded-full mb-1">
                                            <div className={`h-1.5 rounded-full ${m.color}`} style={{width: `${m.score}%`}}></div>
                                        </div>
                                        <div className="text-[10px] text-slate-500">{m.desc}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Generative */}
                            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                                <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase"><BrainCircuit className="w-4 h-4"/> Generative AI</div>
                                {modelLeaderboard.generative.map((m, i) => (
                                    <div key={i} className="bg-slate-950 p-3 rounded border border-slate-800">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-xs font-bold text-white">{m.name}</span>
                                            <span className={`text-xs font-mono font-bold ${m.text}`}>{m.score}%</span>
                                        </div>
                                        <div className="w-full bg-slate-900 h-1.5 rounded-full mb-1">
                                            <div className={`h-1.5 rounded-full ${m.color}`} style={{width: `${m.score}%`}}></div>
                                        </div>
                                        <div className="text-[10px] text-slate-500">{m.desc}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Agentic */}
                            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                                <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase"><Network className="w-4 h-4"/> Agentic Systems</div>
                                {modelLeaderboard.agentic.map((m, i) => (
                                    <div key={i} className="bg-slate-950 p-3 rounded border border-slate-800">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-xs font-bold text-white">{m.name}</span>
                                            <span className={`text-xs font-mono font-bold ${m.text}`}>{m.score}%</span>
                                        </div>
                                        <div className="w-full bg-slate-900 h-1.5 rounded-full mb-1">
                                            <div className={`h-1.5 rounded-full ${m.color}`} style={{width: `${m.score}%`}}></div>
                                        </div>
                                        <div className="text-[10px] text-slate-500">{m.desc}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* System Stats */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                        <h3 className="text-sm font-bold text-white uppercase mb-4 flex items-center gap-2">
                            <Layers className="w-4 h-4 text-slate-400"/> System Statistics
                        </h3>
                        <div className="grid grid-cols-4 gap-4 text-center">
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                <div className="text-[10px] text-slate-500 uppercase mb-1">Queue Pending</div>
                                <div className="text-2xl font-mono text-white">{queueList.filter(q => q.status !== 'Processed').length}</div>
                            </div>
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                <div className="text-[10px] text-slate-500 uppercase mb-1">AI Processed</div>
                                <div className="text-2xl font-mono text-indigo-400">{datasetList.filter(d => d.source !== 'Manual' && d.source !== 'manual_promoted').length}</div>
                            </div>
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                <div className="text-[10px] text-slate-500 uppercase mb-1">Ground Truth</div>
                                <div className="text-2xl font-mono text-emerald-400">{datasetList.filter(d => d.source === 'Manual' || d.source === 'manual_promoted').length}</div>
                            </div>
                            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                <div className="text-[10px] text-slate-500 uppercase mb-1">Profiles</div>
                                <div className="text-2xl font-mono text-sky-400">{profileList.length}</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* PREDICTIVE SANDBOX */}
            {activeTab === 'predictive' && (
                <div className="flex h-full gap-6">
                    <div className="w-1/3 bg-slate-900/50 border border-slate-800 rounded-xl p-6 flex flex-col gap-6">
                        <div>
                            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-2"><FlaskConical className="w-5 h-5"/> Model Sandbox</h2>
                            <p className="text-xs text-slate-400">Train models on the text features of the current Ground Truth dataset.</p>
                        </div>
                        <div className="space-y-4">
                            <div className="p-3 bg-slate-950 rounded border border-slate-800">
                                <label className="text-xs text-slate-500 block mb-2">Algorithm</label>
                                <select value={predictiveModelType} onChange={e => setPredictiveModelType(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-white">
                                    <option value="logistic">Logistic Regression (Simple)</option>
                                    <option value="autogluon">AutoGluon (Gradient Boosting)</option>
                                </select>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-slate-950 rounded border border-slate-800">
                                <span className="text-xs text-slate-300">Use Visual Meta</span>
                                <input type="checkbox" onClick={(e) => runPredictiveTraining(e.currentTarget.checked)} className="accent-indigo-500"/>
                            </div>
                        </div>
                         <div className="mt-4 pt-4 border-t border-slate-800">
                             <div className="text-xs font-bold text-slate-500 uppercase mb-2">Target Schema</div>
                             <pre className="text-[9px] font-mono text-slate-400 bg-black p-2 rounded overflow-x-auto whitespace-pre-wrap">
                                 visual_integrity_score, audio_integrity_score, source_credibility_score, logical_consistency_score, emotional_manipulation_score, video_audio_score, video_caption_score, audio_caption_score, final_veracity_score, tags, stats_likes, stats_shares, stats_comments
                             </pre>
                        </div>
                        <button onClick={() => runPredictiveTraining(false)} className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded font-bold text-xs">Train Baseline</button>
                    </div>
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl p-6 relative overflow-hidden overflow-y-auto">
                        {predictiveResult ? (
                            predictiveResult.status === 'training' ? (
                                <div className="absolute inset-0 flex items-center justify-center text-indigo-400 animate-pulse">Training Model...</div>
                            ) : predictiveResult.error ? ( <div className="text-red-400">{predictiveResult.error}</div> ) : (
                                <div className="space-y-6">
                                    <div className="text-xl font-mono text-white">Training Complete ({predictiveResult.type})</div>
                                    <pre className="text-xs text-slate-400 bg-black p-4 rounded">{JSON.stringify(predictiveResult, null, 2)}</pre>
                                </div>
                            )
                        ) : <div className="flex h-full items-center justify-center text-slate-600">Ready to train.</div>}
                    </div>
                </div>
            )}

            {/* QUEUE TAB */}
            {activeTab === 'queue' && (
                <div className="flex h-full gap-6">
                    <div className="w-[300px] bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 overflow-y-auto">
                        <div className="text-xs font-bold text-indigo-400 uppercase">Config</div>
                        <div className="space-y-1">
                            <label className="text-[10px] text-slate-500">Provider</label>
                            <select value={modelProvider} onChange={e => setModelProvider(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs text-white">
                                <option value="vertex">Vertex AI (Enterprise)</option>
                                <option value="gemini">Gemini API (Public)</option>
                            </select>
                        </div>
                        {modelProvider === 'vertex' ? (
                            <>
                                <div className="space-y-1">
                                    <label className="text-[10px] text-slate-500">Project ID</label>
                                    <input value={projectId} onChange={e => setProjectId(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs" placeholder="gcp-project-id"/>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] text-slate-500">Location</label>
                                    <input value={location} onChange={e => setLocation(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs" placeholder="us-central1"/>
                                </div>
                            </>
                        ) : (
                            <div className="space-y-1">
                                <label className="text-[10px] text-slate-500">API Key</label>
                                <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs" placeholder="AIzaSy..."/>
                            </div>
                        )}
                        <div className="space-y-1">
                            <label className="text-[10px] text-slate-500">Model Name</label>
                            <input value={modelName} onChange={e => setModelName(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs"/>
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] text-slate-500">Reasoning Method</label>
                            <select value={reasoningMethod} onChange={e => setReasoningMethod(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs">
                                <option value="cot">Standard Chain of Thought</option>
                                <option value="fcot">Fractal Chain of Thought</option>
                            </select>
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] text-slate-500">Prompt Persona</label>
                            <select value={promptTemplate} onChange={e => setPromptTemplate(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs">
                                {availablePrompts.length > 0 ? availablePrompts.map(p => (
                                    <option key={p.id} value={p.id}>{p.name}</option>
                                )) : <option value="standard">Standard</option>}
                            </select>
                        </div>
                        <button onClick={startProcessing} disabled={isProcessing} className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded font-bold text-xs flex items-center justify-center gap-2">
                             {isProcessing ? <><RefreshCw className="w-3 h-3 animate-spin"/> Processing...</> : <><Play className="w-3 h-3"/> Start Batch</>}
                        </button>
                    </div>
                    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                        <div className="flex-1 bg-slate-900/30 border border-slate-800 rounded-xl overflow-auto">
                            <table className="w-full text-left text-xs text-slate-400">
                                <thead className="bg-slate-950 sticky top-0"><tr><th className="p-3">Link</th><th className="p-3">Status</th></tr></thead>
                                <tbody>
                                    {queueList.map((q, i) => (
                                        <tr key={i} className="border-t border-slate-800/50 hover:bg-white/5">
                                            <td className="p-3 text-sky-500 font-mono">{q.link}</td>
                                            <td className="p-3">{q.status === 'Processed' ? <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> Done</span> : <span className="text-amber-500">Pending</span>}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        <div ref={logContainerRef} className="h-40 bg-black border border-slate-800 rounded-xl p-3 font-mono text-[10px] text-emerald-500 overflow-y-auto whitespace-pre-wrap">{logs}</div>
                    </div>
                </div>
            )}

            {/* PROFILES TAB */}
            {activeTab === 'profiles' && (
                <div className="flex h-full gap-6">
                    <div className="w-1/3 bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
                        <div className="p-3 bg-slate-950 border-b border-slate-800 text-xs font-bold text-slate-400">Scraped Accounts</div>
                        <div className="flex-1 overflow-auto">
                            {profileList.map((p, i) => (
                                <div key={i} onClick={() => loadProfilePosts(p.username)} 
                                    className={`p-3 border-b border-slate-800/50 cursor-pointer hover:bg-white/5 ${selectedProfile===p.username ? 'bg-indigo-900/20 border-l-2 border-indigo-500' : ''}`}>
                                    <div className="text-sm font-bold text-white">@{p.username}</div>
                                    <div className="text-[10px] text-slate-500">{p.posts_count} posts stored</div>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
                        <div className="p-3 bg-slate-950 border-b border-slate-800 flex justify-between items-center">
                            <span className="text-xs font-bold text-slate-400">Post History {selectedProfile ? `(@${selectedProfile})` : ''}</span>
                            {selectedProfile && (
                                <button onClick={queueUnlabeledPosts} className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[10px] flex items-center gap-1 transition hover:scale-105">
                                    <PlusCircle className="w-3 h-3"/> Auto-Label
                                </button>
                            )}
                        </div>
                        <div className="flex-1 overflow-auto">
                            <table className="w-full text-left text-xs text-slate-400">
                                <thead className="bg-slate-900/80 sticky top-0"><tr><th className="p-3">Date</th><th className="p-3">Text</th><th className="p-3">Status</th><th className="p-3">Action</th></tr></thead>
                                <tbody className="divide-y divide-slate-800">
                                    {profilePosts.map((row, i) => (
                                        <tr key={i} className="hover:bg-white/5">
                                            <td className="p-3 whitespace-nowrap text-slate-500">{row.timestamp?.split('T')[0]}</td>
                                            <td className="p-3 truncate max-w-[300px]">{row.text}</td>
                                            <td className="p-3">{row.is_labeled ? <span className="text-emerald-500 flex items-center gap-1"><ShieldCheck className="w-3 h-3"/> Labeled</span> : <span className="text-slate-600">Unlabeled</span>}</td>
                                            <td className="p-3 flex gap-2">
                                                <button onClick={() => sendToManualLabeler(row.link, row.text)} className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1 bg-indigo-900/30 px-2 py-1 rounded hover:bg-indigo-900/50"><PenTool className="w-3 h-3"/> Manual</button>
                                                <a href={row.link} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white p-1"><ExternalLink className="w-3 h-3"/></a>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* DATASET TAB (Data Manager) */}
            {activeTab === 'dataset' && (
                <div className="h-full overflow-auto bg-slate-900/50 border border-slate-800 rounded-xl flex flex-col">
                    <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950">
                        <span className="font-bold text-slate-400 text-sm flex items-center gap-2">
                            <Archive className="w-4 h-4"/> Data Manager (Unified)
                        </span>
                        <div className="flex gap-2 items-center">
                             <div className="text-[10px] text-slate-500 bg-slate-900 px-2 py-1 rounded border border-slate-800 flex gap-2">
                                 <span>Total: {datasetList.length}</span>
                                 <span>AI: {datasetList.filter(d => d.source === 'AI').length}</span>
                                 <span>Manual: {datasetList.filter(d => d.source === 'Manual').length}</span>
                             </div>
                             <button onClick={promoteSelected} className="bg-emerald-600 text-white text-xs px-3 py-1 rounded font-bold hover:bg-emerald-500 flex items-center gap-2">
                                 <ShieldCheck className="w-3 h-3"/> Add Selected to Ground Truth
                             </button>
                        </div>
                    </div>
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-left text-xs text-slate-400">
                            <thead className="bg-slate-950 sticky top-0">
                                <tr>
                                    <th className="p-3 w-8"><Square className="w-4 h-4 text-slate-600"/></th>
                                    <th className="p-3">Source</th>
                                    <th className="p-3">ID</th>
                                    <th className="p-3">Caption</th>
                                    <th className="p-3">Score</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800">
                                {datasetList.map((row, i) => (
                                    <tr key={i} className={`hover:bg-white/5 ${selectedItems.has(row.id) ? 'bg-indigo-900/20' : ''} ${row.source==='Manual'?'bg-emerald-900/5':''}`}>
                                        <td className="p-3 cursor-pointer" onClick={() => toggleSelection(row.id)}>
                                            {selectedItems.has(row.id) ? <CheckSquare className="w-4 h-4 text-indigo-400"/> : <Square className="w-4 h-4 text-slate-600"/>}
                                        </td>
                                        <td className="p-3">
                                            {row.source === 'Manual' ? (
                                                <span className="text-emerald-400 text-[10px] font-bold border border-emerald-900 bg-emerald-900/20 px-1 rounded">MANUAL</span>
                                            ) : (
                                                <span className="text-indigo-400 text-[10px] font-bold border border-indigo-900 bg-indigo-900/20 px-1 rounded">AI</span>
                                            )}
                                        </td>
                                        <td className="p-3 font-mono text-slate-500">{row.id}</td>
                                        <td className="p-3 truncate max-w-[300px]" title={row.caption}>{row.caption}</td>
                                        <td className="p-3 font-bold text-white">{row.final_veracity_score}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* GROUND TRUTH TAB */}
            {activeTab === 'groundtruth' && (
                <div className="h-full overflow-auto bg-slate-900/50 border border-slate-800 rounded-xl flex flex-col">
                     <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950">
                        <span className="font-bold text-emerald-400 text-sm flex items-center gap-2"><ShieldCheck className="w-4 h-4"/> Verified Ground Truth CSV</span>
                        <div className="flex gap-2">
                             <span className="text-xs text-slate-500 py-1">{datasetList.filter(d => d.source === 'Manual').length} Verified Items</span>
                             <button onClick={deleteSelected} className="bg-red-600 text-white text-xs px-3 py-1 rounded font-bold hover:bg-red-500 flex items-center gap-2">
                                <Trash2 className="w-3 h-3"/> Delete Selected
                             </button>
                        </div>
                    </div>
                     <div className="flex-1 overflow-auto">
                        <table className="w-full text-left text-xs text-slate-400">
                            <thead className="bg-slate-950 sticky top-0">
                                <tr>
                                    <th className="p-3 w-8"><Square className="w-4 h-4 text-slate-600"/></th>
                                    <th className="p-3">ID</th>
                                    <th className="p-3">Caption</th>
                                    <th className="p-3">Score</th>
                                    <th className="p-3">Source</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800">
                                {datasetList.filter(d => d.source === 'Manual').map((row, i) => (
                                    <tr key={i} className={`hover:bg-white/5 ${selectedItems.has(row.id) ? 'bg-red-900/10' : ''}`}>
                                        <td className="p-3 cursor-pointer" onClick={() => toggleSelection(row.id)}>
                                            {selectedItems.has(row.id) ? <CheckSquare className="w-4 h-4 text-red-400"/> : <Square className="w-4 h-4 text-slate-600"/>}
                                        </td>
                                        <td className="p-3 font-mono text-emerald-400">{row.id}</td>
                                        <td className="p-3 truncate max-w-[300px]" title={row.caption}>{row.caption}</td>
                                        <td className="p-3 font-bold text-white">{row.final_veracity_score}</td>
                                        <td className="p-3 text-[10px] uppercase text-slate-500">{row.source}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* MANUAL LABELING STUDIO */}
            {activeTab === 'manual' && (
                <div className="flex h-full gap-6">
                    <div className="w-[280px] bg-slate-900/50 border border-slate-800 rounded-xl flex flex-col overflow-hidden">
                        <div className="flex border-b border-slate-800">
                             <button onClick={() => setLabelBrowserMode('queue')} className={`flex-1 py-3 text-xs font-bold ${labelBrowserMode==='queue'?'text-indigo-400 border-b-2 border-indigo-500':''}`}>Queue</button>
                             <button onClick={() => setLabelBrowserMode('dataset')} className={`flex-1 py-3 text-xs font-bold ${labelBrowserMode==='dataset'?'text-indigo-400 border-b-2 border-indigo-500':''}`}>Reviewed</button>
                        </div>
                        <div className="p-2 border-b border-slate-800">
                             <input value={labelFilter} onChange={e => setLabelFilter(e.target.value)} placeholder="Filter..." className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-white"/>
                        </div>
                        <div className="flex-1 overflow-auto">
                            {(labelBrowserMode==='queue' ? queueList : datasetList)
                                .filter(i => (i.link || '').includes(labelFilter) || (i.id || '').includes(labelFilter))
                                .map((item, i) => (
                                    <div key={i} onClick={() => loadFromBrowser(item, labelBrowserMode)} 
                                        className={`p-3 border-b border-slate-800/50 cursor-pointer hover:bg-white/5 ${manualLink===item.link?'bg-indigo-900/20 border-l-2 border-indigo-500':''}`}>
                                        <div className="text-[10px] text-indigo-400 font-mono mb-1 truncate">{item.id || 'Pending'}</div>
                                        <div className="text-xs text-slate-300 truncate">{item.link}</div>
                                    </div>
                            ))}
                        </div>
                    </div>
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl p-6 overflow-y-auto">
                        <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-800">
                             <h2 className="text-lg font-bold text-white flex items-center gap-2"><PenTool className="w-5 h-5"/> Studio</h2>
                             <div className="flex gap-2">
                                <a href={manualLink} target="_blank" rel="noreferrer" className={`bg-slate-800 text-white px-3 py-2 rounded-lg font-bold flex gap-2 ${!manualLink && 'opacity-50 pointer-events-none'}`}><ExternalLink className="w-4 h-4"/> Open</a>
                                <button onClick={submitManualLabel} className="bg-emerald-600 text-white px-6 py-2 rounded-lg font-bold flex gap-2"><ClipboardCheck className="w-4 h-4"/> Save & Add to GT</button>
                             </div>
                        </div>
                        <div className="space-y-6">
                             <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                 <div className="mb-4">
                                     <label className="text-xs uppercase text-slate-500 font-bold">Link</label>
                                     <input value={manualLink} onChange={e => setManualLink(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-indigo-400 font-mono mt-1"/>
                                 </div>
                                 <div>
                                     <label className="text-xs uppercase text-slate-500 font-bold">Caption</label>
                                     <textarea value={manualCaption} onChange={e => setManualCaption(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-300 mt-1 h-20"/>
                                 </div>
                             </div>
                             
                             <div className="grid grid-cols-2 gap-8">
                                 <div>
                                     <h3 className="text-sm font-bold text-indigo-400 uppercase mb-4 border-b border-slate-800 pb-2">Veracity Vectors</h3>
                                     {['visual', 'audio', 'source', 'logic', 'emotion'].map(k => (
                                         <div key={k} className="mb-4">
                                             <div className="flex justify-between text-xs mb-1">
                                                 <span className="capitalize text-slate-300 font-bold">{k}</span>
                                                 <span className="text-indigo-400 font-mono font-bold">{(manualScores as any)[k]}/10</span>
                                             </div>
                                             <input type="range" min="1" max="10" value={(manualScores as any)[k]} onChange={e => setManualScores({...manualScores, [k]: parseInt(e.target.value)})} className="w-full accent-indigo-500"/>
                                         </div>
                                     ))}
                                 </div>
                                 <div>
                                     <h3 className="text-sm font-bold text-emerald-400 uppercase mb-4 border-b border-slate-800 pb-2">Modality Alignment</h3>
                                     {['va', 'vc', 'ac'].map(k => (
                                         <div key={k} className="mb-4">
                                              <div className="flex justify-between text-xs mb-1">
                                                 <span className="capitalize text-slate-300 font-bold">{k.replace('v', 'Video-').replace('a', 'Audio-').replace('c', 'Caption')}</span>
                                                 <span className="text-emerald-400 font-mono font-bold">{(manualScores as any)[k]}/10</span>
                                             </div>
                                             <input type="range" min="1" max="10" value={(manualScores as any)[k]} onChange={e => setManualScores({...manualScores, [k]: parseInt(e.target.value)})} className="w-full accent-emerald-500"/>
                                         </div>
                                     ))}
                                 </div>
                             </div>

                             <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                                  <div className="flex justify-between items-center mb-4">
                                      <h3 className="text-sm font-bold text-white uppercase">Final Veracity Score</h3>
                                      <span className="text-2xl font-bold font-mono text-emerald-400">{manualScores.final}</span>
                                  </div>
                                  <input type="range" min="0" max="100" value={manualScores.final} onChange={e => setManualScores({...manualScores, final: parseInt(e.target.value)})} className="w-full accent-white mb-6"/>
                                  
                                  <div className="mb-4">
                                     <label className="text-xs uppercase text-slate-500 font-bold">Reasoning</label>
                                     <textarea value={manualReasoning} onChange={e => setManualReasoning(e.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-300 mt-1 h-24" placeholder="Justification..."/>
                                 </div>
                                 <div>
                                     <label className="text-xs uppercase text-slate-500 font-bold">Tags</label>
                                     <input value={manualTags} onChange={handleTagsChange} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm text-slate-300 mt-1" placeholder="politics, satire..."/>
                                     {existingTags.length > 0 && (
                                         <div className="flex flex-wrap gap-2 mt-2">
                                             {existingTags.map(t => (
                                                 <button key={t} onClick={() => toggleTag(t)} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-[10px] rounded text-slate-400 border border-slate-700">{t}</button>
                                             ))}
                                         </div>
                                     )}
                                 </div>
                             </div>
                        </div>
                    </div>
                </div>
            )}

            {/* COMMUNITY TAB */}
            {activeTab === 'community' && (
                <div className="flex h-full gap-6">
                    <div className="w-1/3 bg-slate-900/50 border border-slate-800 rounded-xl overflow-auto">
                        <div className="p-3 bg-slate-950 border-b border-slate-800 text-xs font-bold text-slate-400">Comment Datasets</div>
                        {communityDatasets.map((d, i) => (
                            <div key={i} onClick={() => analyzeComments(d.id)} className="p-4 border-b border-slate-800/50 cursor-pointer hover:bg-white/5">
                                <div className="text-xs font-mono text-indigo-400 mb-1">{d.id}</div>
                                <div className="text-[10px] text-slate-500">{d.count} comments</div>
                            </div>
                        ))}
                    </div>
                    <div className="flex-1 flex flex-col justify-center items-center bg-slate-900/20 border border-slate-800 rounded-xl p-8">
                        {communityAnalysis ? (
                            <div className="text-center w-full max-w-md">
                                <div className="text-xs uppercase text-slate-500 mb-2 tracking-widest">Community Quantization</div>
                                <h2 className="text-5xl font-bold text-white mb-2">{communityAnalysis.trust_score?.toFixed(0)}<span className="text-xl text-slate-600">/100</span></h2>
                                <div className={`text-lg font-bold mb-8 px-4 py-1 rounded-full inline-block ${communityAnalysis.trust_score < 40 ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                                    {communityAnalysis.verdict}
                                </div>
                                <div className="grid grid-cols-2 gap-4 text-left bg-slate-950 p-6 rounded-xl border border-slate-800">
                                    <div className="border-r border-slate-800 pr-4">
                                        <div className="text-[10px] uppercase text-red-500 font-bold mb-1">Skeptical Points</div>
                                        <div className="text-2xl text-white">{communityAnalysis.details?.skeptical_comments}</div>
                                    </div>
                                    <div className="pl-4">
                                        <div className="text-[10px] uppercase text-emerald-500 font-bold mb-1">Trust Points</div>
                                        <div className="text-2xl text-white">{communityAnalysis.details?.trusting_comments}</div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-slate-600 flex flex-col items-center">
                                <MessageSquare className="w-12 h-12 mb-4 opacity-20"/>
                                <span>Select a dataset to analyze community sentiment.</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ANALYTICS TAB */}
            {activeTab === 'analytics' && (
                <div className="h-full overflow-auto">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-bold text-indigo-400 uppercase flex items-center gap-2">
                            <UserCheck className="w-4 h-4"/> Account Integrity Leaderboard
                        </h3>
                        <button onClick={() => setRefreshTrigger(p => p+1)} className="text-xs text-slate-500 hover:text-white flex gap-1 items-center"><RefreshCw className="w-3 h-3"/> Refresh</button>
                    </div>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
                        <table className="w-full text-left text-xs text-slate-400">
                            <thead className="bg-slate-950"><tr><th className="p-4">User</th><th className="p-4">Avg Veracity</th><th className="p-4">Samples</th><th className="p-4">Rating</th></tr></thead>
                            <tbody className="divide-y divide-slate-800">
                                {integrityBoard.map((row, i) => (
                                    <tr key={i} className="hover:bg-white/5">
                                        <td className="p-4 font-bold text-white">@{row.username}</td>
                                        <td className="p-4 text-indigo-300 font-mono text-lg">{row.avg_veracity}</td>
                                        <td className="p-4 text-slate-500">{row.posts_labeled} posts</td>
                                        <td className="p-4">
                                            {row.avg_veracity > 70 ? <span className="text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded font-bold">High Trust</span> : 
                                             row.avg_veracity < 40 ? <span className="text-red-500 bg-red-500/10 px-2 py-1 rounded font-bold">Low Trust</span> :
                                             <span className="text-amber-500 bg-amber-500/10 px-2 py-1 rounded font-bold">Mixed</span>}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

         </div>
      </div>
    </div>
  )
}

export default App
