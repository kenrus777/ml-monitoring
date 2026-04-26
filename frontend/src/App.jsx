import{useState,useEffect,useRef}from'react';import{LineChart,Line,XAxis,YAxis,CartesianGrid,Tooltip,Legend,ResponsiveContainer,ReferenceLine}from'recharts';
const API='',WS=(location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws/live';
const C={green:'#00c896',red:'#ef4444',amber:'#fbbf24',blue:'#818cf8',purple:'#a78bfa',bg:'#060810',bg2:'#0c0e1a',bg3:'#111428',border:'#1e2340',text:'#e2e8f0',muted:'#475569'};
const SC={stable:C.green,warning:C.amber,drift:'#f97316',critical:C.red,healthy:C.green,degraded:C.red};
const useFetch=url=>{const[d,setD]=useState(null),[l,setL]=useState(true);useEffect(()=>{setL(true);fetch(url).then(r=>r.json()).then(d=>{setD(d);setL(false);}).catch(()=>setL(false));},[url]);return{data:d,loading:l};};
const Card=({title,children,style={}})=><div style={{background:C.bg2,border:`1px solid ${C.border}`,borderRadius:12,padding:16,marginBottom:16,...style}}>{title&&<div style={{fontSize:'0.62rem',letterSpacing:'3px',color:C.muted,textTransform:'uppercase',marginBottom:14}}>{title}</div>}{children}</div>;
const Badge=({status})=>{const c=SC[status?.toLowerCase()]||C.muted;return<span style={{background:c+'18',color:c,border:`1px solid ${c}40`,borderRadius:20,padding:'2px 10px',fontSize:'0.62rem',letterSpacing:1,textTransform:'uppercase',fontWeight:700}}>{status}</span>;};
const Stat=({label,value,sub,color})=><div style={{background:C.bg3,border:`1px solid ${C.border}`,borderRadius:10,padding:14,textAlign:'center',borderTop:`2px solid ${color||C.border}`}}><div style={{fontSize:'0.55rem',color:C.muted,letterSpacing:2,textTransform:'uppercase',marginBottom:6}}>{label}</div><div style={{fontSize:'1.4rem',fontWeight:900,color:color||C.text}}>{value??'—'}</div>{sub&&<div style={{fontSize:'0.6rem',color:C.muted,marginTop:4}}>{sub}</div>}</div>;
function Heatmap(){const{data,loading}=useFetch(`${API}/drift/timeline`);if(loading)return<div style={{color:C.muted,fontSize:'0.75rem'}}>Loading...</div>;if(!data)return null;const tl=data.timeline||[];if(!tl.length)return null;const feats=Object.keys(tl[0]?.features||{});const pc=p=>p>0.2?C.red:p>0.1?C.amber:C.green;return<div style={{overflowX:'auto'}}><table style={{borderCollapse:'collapse',width:'100%',fontSize:'0.65rem'}}><thead><tr><th style={{padding:'4px 8px',color:C.muted,textAlign:'left'}}>Feature</th>{tl.map(t=><th key={t.day} style={{padding:'4px 8px',color:C.muted,textAlign:'center'}}>{t.date}</th>)}</tr></thead><tbody>{feats.map(f=><tr key={f}><td style={{padding:'4px 8px',color:C.text,whiteSpace:'nowrap'}}>{f}</td>{tl.map(t=>{const p=t.features[f]?.psi??0;return<td key={t.day} style={{padding:'3px 6px',textAlign:'center'}}><div style={{background:pc(p)+'30',color:pc(p),borderRadius:4,padding:'2px 4px',fontSize:'0.58rem'}}>{p.toFixed(3)}</div></td>;})}</tr>)}</tbody></table><div style={{marginTop:8,display:'flex',gap:12,fontSize:'0.6rem',color:C.muted}}><span style={{color:C.green}}>■ &lt;0.1 Stable</span><span style={{color:C.amber}}>■ 0.1-0.2 Warning</span><span style={{color:C.red}}>■ &gt;0.2 Drift</span></div></div>;}
function PerfChart(){const{data,loading}=useFetch(`${API}/performance/timeline`);if(loading)return<div style={{color:C.muted,fontSize:'0.75rem'}}>Loading...</div>;if(!data)return null;return<ResponsiveContainer width='100%' height={200}><LineChart data={data.snapshots||[]} margin={{top:5,right:20,left:0,bottom:5}}><CartesianGrid strokeDasharray='3 3' stroke={C.border}/><XAxis dataKey='date' tick={{fill:C.muted,fontSize:10}}/><YAxis domain={[0.8,1.0]} tick={{fill:C.muted,fontSize:10}}/><Tooltip contentStyle={{background:C.bg2,border:`1px solid ${C.border}`,borderRadius:8}} labelStyle={{color:C.text}}/><Legend wrapperStyle={{fontSize:'0.65rem'}}/><ReferenceLine y={data.baseline?.auc_roc} stroke={C.blue} strokeDasharray='4 4'/><Line type='monotone' dataKey='auc_roc' stroke={C.green} dot={false} name='AUC-ROC' strokeWidth={2}/><Line type='monotone' dataKey='f1_score' stroke={C.amber} dot={false} name='F1' strokeWidth={2}/><Line type='monotone' dataKey='precision' stroke={C.blue} dot={false} name='Precision' strokeWidth={1.5} strokeDasharray='3 3'/><Line type='monotone' dataKey='recall' stroke={C.purple} dot={false} name='Recall' strokeWidth={1.5} strokeDasharray='3 3'/></LineChart></ResponsiveContainer>;}
function Live(){const[evs,setEvs]=useState([]);const ws=useRef(null);useEffect(()=>{try{ws.current=new WebSocket(WS);ws.current.onmessage=e=>setEvs(p=>[JSON.parse(e.data),...p].slice(0,20));}catch(e){}return()=>ws.current?.close();},[]);const rc={LOW:C.green,MEDIUM:C.amber,HIGH:'#f97316',CRITICAL:C.red};return<div style={{maxHeight:260,overflowY:'auto'}}>{!evs.length&&<div style={{color:C.muted,fontSize:'0.7rem'}}>Connecting...</div>}{evs.map((e,i)=><div key={i} style={{display:'flex',alignItems:'center',gap:8,padding:'6px 0',borderBottom:`1px solid ${C.border}`,fontSize:'0.65rem'}}><div style={{width:8,height:8,borderRadius:'50%',background:rc[e.risk_level]||C.muted,flexShrink:0}}/><div style={{flex:1,color:C.muted}}>{e.transaction_id}</div><div style={{color:C.text}}>{(e.score*100).toFixed(1)}%</div><div style={{color:rc[e.risk_level],background:rc[e.risk_level]+'18',padding:'1px 7px',borderRadius:4,fontSize:'0.55rem'}}>{e.risk_level}</div><div style={{color:C.muted,fontSize:'0.55rem'}}>{e.latency_ms}ms</div></div>)}</div>;}
function Alerts(){const{data,loading}=useFetch(`${API}/alerts`);if(loading)return<div style={{color:C.muted,fontSize:'0.75rem'}}>Loading...</div>;if(!data?.length)return<div style={{color:C.green,fontSize:'0.75rem'}}>✓ No active alerts</div>;const sc={warning:C.amber,critical:C.red};return<div>{data.map((a,i)=><div key={i} style={{background:C.bg3,borderLeft:`3px solid ${sc[a.severity]||C.border}`,borderRadius:8,padding:'10px 12px',marginBottom:8}}><div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}><div style={{fontWeight:700,fontSize:'0.78rem'}}>{a.title}</div><Badge status={a.severity}/></div><div style={{fontSize:'0.68rem',color:C.muted}}>{a.description}</div></div>)}</div>;}
export default function App(){
  const{data:s,loading}=useFetch(`${API}/summary`);
  const[tab,setTab]=useState('overview');
  const tabs=['overview','drift','performance','alerts','live feed'];
  return<div style={{background:C.bg,minHeight:'100vh',color:C.text,fontFamily:'Syne,sans-serif'}}>
    <div style={{background:C.bg2,borderBottom:`1px solid ${C.border}`,padding:'10px 20px',display:'flex',alignItems:'center',justifyContent:'space-between',position:'sticky',top:0,zIndex:10}}>
      <div style={{fontSize:'1rem',fontWeight:900,color:C.green,letterSpacing:3}}>ML MONITOR <span style={{color:C.muted}}>// DASHBOARD</span></div>
      <div style={{display:'flex',gap:8,alignItems:'center'}}>{s&&<Badge status={s.model_status}/>}<div style={{fontSize:'0.62rem',color:C.muted}}>{s?.model_version}</div></div>
    </div>
    <div style={{background:C.bg3,borderBottom:`1px solid ${C.border}`,padding:'0 20px',display:'flex',gap:4}}>
      {tabs.map(t=><button key={t} onClick={()=>setTab(t)} style={{padding:'10px 16px',border:'none',borderBottom:tab===t?`2px solid ${C.green}`:'2px solid transparent',background:'transparent',color:tab===t?C.green:C.muted,fontFamily:'Syne,sans-serif',fontWeight:700,fontSize:'0.75rem',cursor:'pointer',textTransform:'capitalize'}}>{t}</button>)}
    </div>
    <div style={{maxWidth:1200,margin:'0 auto',padding:20}}>
      {tab==='overview'&&<>
        {!loading&&s&&<div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:12,marginBottom:20}}>
          <Stat label='Model Status' value={s.model_status?.toUpperCase()} color={SC[s.model_status]}/>
          <Stat label='Drift Status' value={s.drift_status?.toUpperCase()} color={SC[s.drift_status]}/>
          <Stat label='AUC-ROC 7d' value={s.auc_7d?.toFixed(3)||'—'} sub={`Baseline: ${s.auc_baseline?.toFixed(3)}`} color={C.green}/>
          <Stat label='Active Alerts' value={s.active_alerts} color={s.active_alerts>0?C.red:C.green}/>
          <Stat label='Predictions' value={s.predictions_today?.toLocaleString()} color={C.blue}/>
          <Stat label='P99 Latency' value={`${s.p99_latency_ms?.toFixed(1)}ms`} color={C.amber}/>
          <Stat label='Days Since Retrain' value={s.days_since_retrain} color={s.days_since_retrain>30?C.red:C.green}/>
          <Stat label='Drifted Features' value={s.drifted_features} color={s.drifted_features>2?C.red:s.drifted_features>0?C.amber:C.green}/>
        </div>}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <Card title='Performance Timeline'><PerfChart/></Card>
          <Card title='Active Alerts'><Alerts/></Card>
        </div>
      </>}
      {tab==='drift'&&<Card title='PSI Heatmap — Feature Drift Over 30 Days'><Heatmap/></Card>}
      {tab==='performance'&&<Card title='Model Performance Over Time'><PerfChart/></Card>}
      {tab==='alerts'&&<Card title='Active Monitoring Alerts'><Alerts/></Card>}
      {tab==='live feed'&&<Card title='Live Prediction Stream (WebSocket)'><Live/></Card>}
    </div>
  </div>;
}
