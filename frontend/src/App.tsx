import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, Play, Upload, List, Database, BarChart2, ExternalLink, 
  Users, MessageSquare, TrendingUp, ShieldCheck, UserCheck, Search, PlusCircle, 
  StopCircle, RefreshCw, CheckCircle2, PenTool, ClipboardCheck, Info, Clock, FileText,
  Tag, Home, Cpu, FlaskConical, Target, Trash2, ArrowUpRight, CheckSquare, Square,
  Layers, Activity, Zap, BrainCircuit, Network, Archive, Plus, Edit3, RotateCcw,
  Bot, Trophy, HelpCircle, Settings, Calculator
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
  const [customQuery, setCustomQuery] = useState('');
  const [maxRetries, setMaxRetries] = useState(1);
  const [availablePrompts, setAvailablePrompts] = useState<any[]>([]);

  // Predictive Config
  const [predictiveModelType, setPredictiveModelType] = useState('logistic');
  const [predictiveResult, setPredictiveResult] = useState<any>(null);

  // Data States
  const [queueList, setQueueList] = useState<any[]>([]);
  const [selectedQueueItems, setSelectedQueueItems] = useState<Set<string>>(new Set());
  const [singleLinkInput, setSingleLinkInput] = useState(''); 
  const [profileList, setProfileList] = useState<any[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<any>(null);
  const [profilePosts, setProfilePosts] = useState<any[]>([]);
  const [communityDatasets, setCommunityDatasets] = useState<any[]>([]);
  const [communityAnalysis, setCommunityAnalysis] = useState<any>(null);
  const [integrityBoard, setIntegrityBoard] = useState<any[]>([]);
  const [datasetList, setDatasetList] = useState<any[]>([]);
  const [benchmarks, setBenchmarks] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<any[]>([]); 
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Tags
  const [configuredTags, setConfiguredTags] = useState<any>({});
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
  const [showRubric, setShowRubric] = useState(false);
  const [aiReference, setAiReference] = useState<any>(null);
  const [labelBrowserMode, setLabelBrowserMode] = useState<'queue' | 'dataset'>('queue');
  const [labelFilter, setLabelFilter] = useState('');

  // Agent Chat State
  const [agentInput, setAgentInput] = useState('');
  const [agentMessages, setAgentMessages] = useState<any[]>([]);
  const [agentThinking, setAgentThinking] = useState(false);
  const [agentEndpoint, setAgentEndpoint] = useState('/a2a');
  const [agentMethod, setAgentMethod] = useState('agent.process');

  const handleTagsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setManualTags(e.target.value);
  };

  useEffect(() => {
    const load = async (url: string, setter: any) => {
        try { const res = await fetch(url); const d = await res.json(); setter(Array.isArray(d) ? d : (d.status==='no_data'?null:d)); } catch(e) {}
    };

    load('/config/prompts', setAvailablePrompts);
    load('/config/tags', setConfiguredTags);

    if (activeTab === 'home') {
        load('/benchmarks/stats', setBenchmarks);
        load('/benchmarks/leaderboard', setLeaderboard);
    }
    if (activeTab === 'queue') {
        load('/queue/list', setQueueList);
        setSelectedQueueItems(new Set());
    }
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

  const existingTags = React.useMemo(() => {
    const tags = new Set<string>();
    Object.keys(configuredTags).forEach(t => tags.add(t));
    if (datasetList && Array.isArray(datasetList)) {
        datasetList.forEach(item => {
            if (item.tags && typeof item.tags === 'string') {
                item.tags.split(/[,\n|;]+/).forEach((t: string) => {
                    const clean = t.trim().replace(/^['"]|['"]$/g, '');
                    if (clean.length > 1) tags.add(clean);
                });
            }
        });
    }
    return Array.from(tags).sort();
  }, [datasetList, configuredTags]);

  const toggleTag = (tag: string) => {
    let current = manualTags.split(',').map(t => t.trim()).filter(Boolean);
    if (current.includes(tag)) {
        current = current.filter(t => t !== tag);
    } else {
        current.push(tag);
    }
    setManualTags(current.join(', '));
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
      setAiReference(null); 
      
      const ref = datasetList.find(d => 
          d.source !== 'Manual' && 
          d.source !== 'manual_promoted' && 
          (d.link === link)
      );
      setAiReference(ref || null);
      
      setActiveTab('manual');
  };
  
  const loadFromBrowser = (item: any, mode: 'queue' | 'dataset') => {
      setManualLink(item.link);
      
      const ref = datasetList.find(d => 
          d.source !== 'Manual' && 
          d.source !== 'manual_promoted' && 
          (d.id === item.id || d.link === item.link)
      );
      setAiReference(ref || null);

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

  const editSelectedLabel = () => {
      if(selectedItems.size !== 1) return alert("Please select exactly one item to edit.");
      const id = Array.from(selectedItems)[0];
      const item = datasetList.find(d => d.id === id);
      if(!item) return;
      loadFromBrowser(item, 'dataset');
      setActiveTab('manual');
  };

  const toggleSelection = (id: string) => {
      const newSet = new Set(selectedItems);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      setSelectedItems(newSet);
  };
  
  const toggleQueueSelection = (link: string) => {
      const newSet = new Set(selectedQueueItems);
      if (newSet.has(link)) newSet.delete(link);
      else newSet.add(link);
      setSelectedQueueItems(newSet);
  };

  const promoteSelected = async () => {
      if (selectedItems.size === 0) return alert("No items selected.");
      if (!confirm(`Promote ${selectedItems.size} items to Ground Truth?`)) return;
      try {
          const res = await fetch('/manual/promote', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ ids: Array.from(selectedItems) })
          });
          const d = await res.json();
          if(d.status === 'success') {
              alert(`Successfully promoted ${d.promoted_count} items.`);
              setSelectedItems(new Set());
              setRefreshTrigger(p => p+1);
          } else alert("Promotion failed: " + d.message);
      } catch(e: any) { alert("Network error: " + e.toString()); }
  };

  const verifySelected = async () => {
      if (selectedItems.size === 0) return alert("No items selected.");
      if (!confirm(`Queue ${selectedItems.size} Ground Truth items for AI Verification?`)) return;
      try {
          const res = await fetch('/manual/verify_queue', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ ids: Array.from(selectedItems) })
          });
          const d = await res.json();
          if(d.status === 'success') {
              alert(d.message);
              setSelectedItems(new Set());
          } else alert("Error: " + d.message);
      } catch(e) { alert("Network error."); }
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

  const deleteDataEntries = async () => {
      if (selectedItems.size === 0) return alert("No items selected.");
      if (!confirm(`Delete ${selectedItems.size} items? This cannot be undone.`)) return;

      const selectedArray = Array.from(selectedItems);
      const manualIds = selectedArray.filter(id => datasetList.find(d => d.id === id)?.source === 'Manual');
      const aiIds = selectedArray.filter(id => {
          const item = datasetList.find(d => d.id === id);
          return item?.source === 'AI' || !item?.source; 
      });

      try {
          let msg = "";
          if (manualIds.length > 0) {
              const res = await fetch('/manual/delete', {
                  method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({ ids: manualIds })
              });
              const d = await res.json();
              if (d.status === 'success') msg += `Deleted ${d.deleted_count} Manual items. `;
              else msg += `Failed Manual delete: ${d.message}. `;
          }

          if (aiIds.length > 0) {
              const res = await fetch('/dataset/delete', {
                  method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({ ids: aiIds })
              });
              const d = await res.json();
              if (d.status === 'success') msg += `Deleted ${d.deleted_count} AI items. `;
              else msg += `Failed AI delete: ${d.message}. `;
          }

          alert(msg || "Done.");
          setSelectedItems(new Set());
          setRefreshTrigger(p => p + 1);
      } catch (e) {
          alert("Network error: " + e);
      }
  };

  const submitManualLabel = async () => {
      if(!manualLink) return alert("Link is required.");
      const payload = {
          link: manualLink, caption: manualCaption, tags: manualTags, reasoning: manualReasoning,
          visual_integrity_score: manualScores.visual, audio_integrity_score: manualScores.audio,
          source_credibility_score: manualScores.source, logical_consistency_score: manualScores.logic,
          emotional_manipulation_score: manualScores.emotion,
          video_audio_score: manualScores.va, video_caption_score: manualScores.vc, audio_caption_score: manualScores.ac,
          final_veracity_score: manualScores.final,
          classification: "Manual Verified"
      };
      try {
          const res = await fetch('/manual/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
          const d = await res.json();
          if(d.status === 'success') { 
              alert("Label Saved! Data updated."); 
              setRefreshTrigger(p => p+1); 
          } else alert("Error saving label: " + d.message);
      } catch(e: any) { alert("Network error: " + e.toString()); }
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
          const data = await res.json();
          setPredictiveResult(data);
          setRefreshTrigger(p => p+1);
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

  const addSingleLink = async () => {
      if(!singleLinkInput) return;
      try {
          const res = await fetch('/queue/add', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ link: singleLinkInput })
          });
          const d = await res.json();
          if(d.status === 'success') {
              setSingleLinkInput('');
              setRefreshTrigger(p => p+1);
          } else { alert(d.message); }
      } catch(e) { alert("Error adding link"); }
  };

  const clearProcessed = async () => {
      if(!confirm("Remove all 'Processed' items from the queue?")) return;
      try {
          const res = await fetch('/queue/clear_processed', { method: 'POST' });
          const d = await res.json();
          alert(`Removed ${d.removed_count} processed items.`);
          setRefreshTrigger(p => p+1);
      } catch(e) { alert("Error clearing queue."); }
  };
  
  const deleteQueueItems = async () => {
      if(selectedQueueItems.size === 0) return alert("No items selected.");
      if(!confirm(`Remove ${selectedQueueItems.size} items from queue?`)) return;
      try {
          const res = await fetch('/queue/delete', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ links: Array.from(selectedQueueItems) })
          });
          const d = await res.json();
          if(d.status === 'success') {
              alert(`Removed ${d.count} items.`);
              setSelectedQueueItems(new Set());
              setRefreshTrigger(p => p+1);
          }
      } catch(e) { alert("Error deleting queue items."); }
  };

  const stopProcessing = async () => {
      if(!confirm("Stop batch processing?")) return;
      await fetch('/queue/stop', { method: 'POST' });
      setLogs(prev => prev + '\n[SYSTEM] Stop Signal Sent.\n');
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
      fd.append('prompt_template', promptTemplate); fd.append('custom_query', customQuery);
      fd.append('max_reprompts', maxRetries.toString());

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

  const callAgent = async (method: string, payloadParams: any) => {
      return fetch(agentEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
              jsonrpc: "2.0",
              method: method, 
              params: payloadParams,
              id: Date.now()
          })
      });
  };

  const sendAgentMessage = async () => {
      if (!agentInput.trim() || agentThinking) return;
      setAgentMessages(prev => [...prev, {role: 'user', content: agentInput}]);
      const currentInput = agentInput;
      setAgentInput('');
      setAgentThinking(true);
      
      try {
          let res = await callAgent(agentMethod, { input: currentInput });
          let data = await res.json();

          if (data.error && data.error.code === -32601 && agentMethod === 'agent.process') {
              res = await callAgent('agent.generate', { input: currentInput });
              data = await res.json();
              if (!data.error) {
                  setAgentMethod('agent.generate'); 
              }
          }

          let reply = "Agent sent no text.";
          if (data.error) {
              reply = `Agent Error: ${data.error.message || JSON.stringify(data.error)}`;
          } else if (data.result) {
               if (typeof data.result === 'string') reply = data.result;
               else if (data.result.text) reply = data.result.text;
               else if (data.result.content) reply = data.result.content;
               else reply = JSON.stringify(data.result);
          }

          setAgentMessages(prev => [...prev, {role: 'agent', content: reply}]);
          if (currentInput.toLowerCase().includes("queue") || currentInput.includes("http")) {
              setTimeout(() => setRefreshTrigger(p => p+1), 2000);
          }

      } catch (e: any) {
          setAgentMessages(prev => [...prev, {role: 'agent', content: `Connection Error: ${e.message}.`}]);
      } finally {
          setAgentThinking(false);
      }
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
               {id:'agent', l:'Agent Nexus', i:Bot},
               {id:'predictive', l:'Predictive Sandbox', i:FlaskConical},
               {id:'queue', l:'Ingest Queue', i:List}, 
               {id:'profiles', l:'User Profiles', i:Users}, 
               {id:'manual', l:'Labeling Studio', i:PenTool},
               {id:'dataset', l:'Data Manager', i:Archive}, 
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
                            <div className="grid grid-cols-2 gap-4 mt-6">
                                <div className="p-4 bg-black/50 border border-slate-800 rounded-lg">
                                    <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase mb-2">
                                        <Calculator className="w-3 h-3"/> Post Veracity Score
                                    </div>
                                    <div className="text-[10px] font-mono text-slate-400">
                                        Weighted Average of Vectors:<br/>
                                        <code>Score = Σ(Visual, Audio, Logic, Source) / N</code><br/>
                                        <span className="text-slate-500 italic">Determined by Agent Reasoning</span>
                                    </div>
                                </div>
                                <div className="p-4 bg-black/50 border border-slate-800 rounded-lg">
                                    <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase mb-2">
                                        <Target className="w-3 h-3"/> Config Accuracy
                                    </div>
                                    <div className="text-[10px] font-mono text-slate-400">
                                        Delta from Ground Truth:<br/>
                                        <code>Acc % = 100 - |GT_Score - AI_Score|</code><br/>
                                        <span className="text-slate-500 italic">Across all verified samples</span>
                                    </div>
                                </div>
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
                    
                    {/* Configuration Leaderboard */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                        <h3 className="text-sm font-bold text-white uppercase mb-4 flex items-center gap-2">
                            <Trophy className="w-4 h-4 text-amber-400"/> Configuration Leaderboard
                        </h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-xs text-slate-400">
                                <thead className="bg-slate-950 text-slate-500 uppercase">
                                    <tr>
                                        <th className="p-3">Type</th>
                                        <th className="p-3">Model</th>
                                        <th className="p-3">Prompt</th>
                                        <th className="p-3">Reasoning</th>
                                        <th className="p-3 text-right">Accuracy</th>
                                        <th className="p-3 text-right">MAE</th>
                                        <th className="p-3 text-right">Samples</th>
                                        <th className="p-3"></th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800">
                                    {leaderboard && leaderboard.map((row, i) => (
                                        <tr key={i} className="hover:bg-white/5">
                                            <td className="p-3 font-mono text-xs">{row.type === 'GenAI' ? <span className="text-indigo-400">GenAI</span> : <span className="text-pink-400">Pred</span>}</td>
                                            <td className="p-3 font-mono text-white">{row.model}</td>
                                            <td className="p-3">{row.prompt}</td>
                                            <td className="p-3 uppercase text-[10px]">{row.reasoning}</td>
                                            <td className="p-3 text-right font-bold text-emerald-400">{row.accuracy}%</td>
                                            <td className="p-3 text-right">{row.mae}</td>
                                            <td className="p-3 text-right text-slate-500">{row.samples}</td>
                                            <td className="p-3 text-center" title={row.params}>
                                                <div className="group relative">
                                                    <HelpCircle className="w-4 h-4 text-slate-600 cursor-help"/>
                                                    <div className="absolute right-0 bottom-6 w-64 p-3 bg-black border border-slate-700 rounded shadow-xl hidden group-hover:block z-50 text-[10px] whitespace-pre-wrap">
                                                        <div className="font-bold mb-1 text-slate-400">Config Params</div>
                                                        {row.params}
                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                    {(!leaderboard || leaderboard.length === 0) && (
                                        <tr><td colSpan={8} className="p-4 text-center text-slate-600">No benchmark data available.</td></tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* AGENT NEXUS TAB */}
            {activeTab === 'agent' && (
                <div className="flex h-full gap-6">
                    <div className="w-1/3 bg-slate-900/50 border border-slate-800 rounded-xl p-6 flex flex-col">
                        <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                            <BrainCircuit className="w-5 h-5 text-indigo-400"/> Agent Configuration
                        </h2>
                        <div className="text-xs text-slate-400 mb-6">
                            This interface interacts with the <strong>LiarMP4 Agent</strong> running on the Google Cloud Agent Development Kit (ADK) via the A2A Protocol.
                        </div>
                         {/* CONFIGURABLE ENDPOINT */}
                        <div className="bg-slate-950 p-4 rounded border border-slate-800">
                             <div className="text-[10px] uppercase text-slate-500 font-bold mb-2 flex items-center gap-2"><Settings className="w-3 h-3"/> Connection</div>
                             <div className="space-y-2">
                                 <label className="text-[10px] text-slate-400">Agent Endpoint URL</label>
                                 <input 
                                     value={agentEndpoint}
                                     onChange={e => setAgentEndpoint(e.target.value)}
                                     className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-white font-mono placeholder-slate-600"
                                     placeholder="e.g. /a2a or http://localhost:8006/a2a"
                                 />
                                 <div className="text-[9px] text-slate-500">
                                     If proxy fails, try direct backend port: <code>http://localhost:8006/a2a</code>
                                 </div>
                             </div>
                        </div>
                    </div>
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl flex flex-col overflow-hidden">
                        <div className="p-4 border-b border-slate-800 bg-slate-950/50">
                            <div className="text-xs font-bold text-white">Agent Interaction (A2A)</div>
                        </div>
                        <div className="flex-1 p-4 overflow-y-auto space-y-4">
                            {agentMessages.length === 0 && (
                                <div className="text-center text-slate-600 mt-20">
                                    <Bot className="w-12 h-12 mx-auto mb-2 opacity-20"/>
                                    <div>Ready to assist. Paste a link to analyze.</div>
                                </div>
                            )}
                            {agentMessages.map((m, i) => (
                                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] p-3 rounded-lg text-xs ${m.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-300'}`}>
                                        {m.content}
                                    </div>
                                </div>
                            ))}
                            {agentThinking && (
                                <div className="flex justify-start">
                                    <div className="max-w-[80%] p-3 rounded-lg text-xs bg-slate-800 text-slate-300 animate-pulse">
                                        Processing request...
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="p-4 bg-slate-950 border-t border-slate-800 flex gap-2">
                            <input 
                                value={agentInput}
                                onChange={e => setAgentInput(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && sendAgentMessage()}
                                className="flex-1 bg-slate-900 border border-slate-700 rounded p-2 text-xs text-white placeholder-slate-500"
                                placeholder="Message the agent (e.g., 'Analyze this video: https://...')"
                                disabled={agentThinking}
                            />
                            <button onClick={sendAgentMessage} disabled={agentThinking} className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white p-2 rounded">
                                <ArrowUpRight className="w-4 h-4"/>
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* PREDICTIVE SANDBOX */}
            {activeTab === 'predictive' && (
                <div className="flex h-full gap-6">
                    <div className="w-1/3 bg-slate-900/50 border border-slate-800 rounded-xl p-6 flex flex-col gap-6">
                        <div>
                            <h2 className="text-lg font-bold text-white flex items-center gap-2"><FlaskConical className="w-5 h-5"/> Model Sandbox</h2>
                            <p className="text-xs text-slate-400">Train models on the text features of the current Ground Truth dataset.</p>
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
                         <div className="bg-slate-950 border border-slate-800 rounded p-3">
                            <label className="text-[10px] text-slate-500 uppercase font-bold mb-2 block">Quick Ingest</label>
                            <div className="flex gap-2">
                                <input 
                                    value={singleLinkInput} 
                                    onChange={e => setSingleLinkInput(e.target.value)} 
                                    className="flex-1 bg-slate-900 border border-slate-700 rounded p-1.5 text-xs text-white placeholder-slate-600"
                                    placeholder="https://x.com/..."
                                />
                                <button onClick={addSingleLink} className="bg-indigo-600 hover:bg-indigo-500 text-white rounded p-1.5">
                                    <Plus className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

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
                        
                        {/* Process Controls */}
                        {isProcessing ? (
                            <button onClick={stopProcessing} className="w-full py-2 bg-red-600 hover:bg-red-500 text-white rounded font-bold text-xs flex items-center justify-center gap-2 animate-pulse">
                                 <StopCircle className="w-3 h-3"/> STOP PROCESSING
                            </button>
                        ) : (
                            <button onClick={startProcessing} className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded font-bold text-xs flex items-center justify-center gap-2">
                                 <Play className="w-3 h-3"/> Start Batch
                            </button>
                        )}
                        
                        <div className="flex gap-2">
                            <button onClick={clearProcessed} className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded font-bold text-[10px] flex items-center justify-center gap-1">
                                 <Trash2 className="w-3 h-3"/> Clear Done
                            </button>
                            <button onClick={deleteQueueItems} className="flex-1 py-2 bg-red-900/50 hover:bg-red-900 text-red-300 border border-red-900 rounded font-bold text-[10px] flex items-center justify-center gap-1">
                                 <Trash2 className="w-3 h-3"/> Delete Sel
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                        <div className="flex-1 bg-slate-900/30 border border-slate-800 rounded-xl overflow-auto">
                            <table className="w-full text-left text-xs text-slate-400">
                                <thead className="bg-slate-950 sticky top-0">
                                    <tr>
                                        <th className="p-3 w-8"><Square className="w-4 h-4 text-slate-600"/></th>
                                        <th className="p-3">Link</th>
                                        <th className="p-3">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {queueList.map((q, i) => (
                                        <tr key={i} className={`border-t border-slate-800/50 hover:bg-white/5 ${selectedQueueItems.has(q.link) ? 'bg-indigo-900/20' : ''}`}>
                                            <td className="p-3 cursor-pointer" onClick={() => toggleQueueSelection(q.link)}>
                                                {selectedQueueItems.has(q.link) ? <CheckSquare className="w-4 h-4 text-indigo-400"/> : <Square className="w-4 h-4 text-slate-600"/>}
                                            </td>
                                            <td className="p-3 text-sky-500 font-mono break-all">{q.link}</td>
                                            <td className="p-3">
                                                {q.status === 'Processed' ? 
                                                    <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> Done</span> : 
                                                    q.status === 'Error' ? 
                                                    <span className="text-red-500 flex items-center gap-1"><AlertCircle className="w-3 h-3"/> Error</span> :
                                                    <span className="text-amber-500">Pending</span>
                                                }
                                            </td>
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
                             {selectedItems.size === 1 && (
                                <button onClick={editSelectedLabel} className="bg-sky-600 text-white text-xs px-3 py-1 rounded font-bold hover:bg-sky-500 flex items-center gap-2">
                                     <Edit3 className="w-3 h-3"/> Edit Label
                                </button>
                             )}
                             <button onClick={deleteDataEntries} className="bg-red-600 text-white text-xs px-3 py-1 rounded font-bold hover:bg-red-500 flex items-center gap-2">
                                <Trash2 className="w-3 h-3"/> Delete Selected
                             </button>
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
                             <span className="text-xs text-slate-500 py-1 mr-4">{datasetList.filter(d => d.source === 'Manual').length} Verified Items</span>
                             <button onClick={verifySelected} className="bg-indigo-600 text-white text-xs px-3 py-1 rounded font-bold hover:bg-indigo-500 flex items-center gap-2">
                                <RotateCcw className="w-3 h-3"/> Verify Scores (Re-Queue)
                             </button>
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
                <div className="flex h-full gap-6 relative">
                    {/* Rubric Overlay */}
                    {showRubric && (
                        <div className="absolute inset-0 z-50 bg-[#09090b]/95 backdrop-blur-sm p-8 flex justify-center items-center">
                            <div className="bg-slate-900 border border-slate-800 w-full max-w-4xl h-[90vh] rounded-xl flex flex-col shadow-2xl">
                                <div className="p-6 border-b border-slate-800 flex justify-between items-center">
                                    <h2 className="text-xl font-bold text-white">Labeling Guide & Rubric</h2>
                                    <button onClick={() => setShowRubric(false)} className="text-slate-400 hover:text-white"><Trash2 className="w-6 h-6 rotate-45"/></button>
                                </div>
                                <div className="flex-1 overflow-y-auto p-8 prose prose-invert max-w-none">
                                    <h3>Core Scoring Philosophy</h3>
                                    <p><strong>1</strong> = Malicious/Fabricated. <strong>5</strong> = Unknown/Generic. <strong>10</strong> = Authentic/Verified.</p>
                                    <h4>A. Visual Integrity</h4>
                                    <ul>
                                        <li><strong>1-2 (Deepfake):</strong> AI-generated or spatially altered.</li>
                                        <li><strong>3-4 (Deceptive):</strong> Real footage, misleading edit (speed/crop).</li>
                                        <li><strong>5-6 (Context):</strong> Real footage, false context or stock B-roll.</li>
                                        <li><strong>9-10 (Raw):</strong> Verified raw footage/metadata.</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    )}

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
                    
                    {/* Main Workspace with Split View Support */}
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden flex">
                        
                        {/* THE FORM */}
                        <div className="flex-1 p-6 overflow-y-auto">
                            <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-800">
                                 <h2 className="text-lg font-bold text-white flex items-center gap-2"><PenTool className="w-5 h-5"/> Studio</h2>
                                 <div className="flex gap-2">
                                    <button onClick={() => setShowRubric(true)} className="bg-slate-800 hover:bg-slate-700 text-indigo-400 px-3 py-2 rounded-lg font-bold text-xs flex gap-2 items-center"><Info className="w-4 h-4"/> Reference Guide</button>
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

                        {/* AI REFERENCE PANEL */}
                        {aiReference && (
                            <div className="w-[300px] bg-slate-950 border-l border-slate-800 p-4 overflow-y-auto">
                                <h3 className="text-xs font-bold text-indigo-400 uppercase mb-4 flex items-center gap-2">
                                    <BrainCircuit className="w-4 h-4"/> AI Reference
                                </h3>
                                <div className="text-[10px] text-slate-500 mb-2 font-mono break-all">ID: {aiReference.id}</div>
                                
                                <div className="mb-6 bg-slate-900 p-3 rounded border border-slate-800">
                                    <div className="text-xs text-slate-400 font-bold uppercase mb-1">AI Score</div>
                                    <div className={`text-2xl font-mono font-bold ${aiReference.final_veracity_score < 50 ? 'text-red-400' : 'text-emerald-400'}`}>
                                        {aiReference.final_veracity_score}/100
                                    </div>
                                </div>
                                
                                <div className="mb-4">
                                    <div className="text-xs text-slate-400 font-bold uppercase mb-1">Reasoning</div>
                                    <div className="text-xs text-slate-400 whitespace-pre-wrap leading-relaxed p-2 bg-slate-900 rounded border border-slate-800/50">
                                        {aiReference.reasoning || "No reasoning provided."}
                                    </div>
                                </div>

                                <div className="mb-4">
                                    <div className="text-xs text-slate-400 font-bold uppercase mb-1">Configuration</div>
                                    <div className="text-[10px] space-y-1">
                                        <div className="flex justify-between"><span className="text-slate-500">Model:</span> <span className="text-slate-300">{aiReference.config_model || 'Unknown'}</span></div>
                                        <div className="flex justify-between"><span className="text-slate-500">Prompt:</span> <span className="text-slate-300">{aiReference.config_prompt || 'Unknown'}</span></div>
                                        <div className="flex justify-between"><span className="text-slate-500">Reasoning:</span> <span className="text-slate-300">{aiReference.config_reasoning || 'Unknown'}</span></div>
                                    </div>
                                </div>

                                {aiReference.raw_toon && (
                                    <div className="mt-4 pt-4 border-t border-slate-800">
                                        <details>
                                            <summary className="text-[10px] cursor-pointer text-indigo-400 hover:text-indigo-300 font-bold">Show Raw TOON Output</summary>
                                            <pre className="text-[9px] text-slate-500 whitespace-pre-wrap mt-2 bg-black p-2 rounded border border-slate-800 overflow-x-auto">
                                                {aiReference.raw_toon}
                                            </pre>
                                        </details>
                                    </div>
                                )}
                            </div>
                        )}

                    </div>
                </div>
            )}

            {/* COMMUNITY AND ANALYTICS TABS (UNCHANGED) */}
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
