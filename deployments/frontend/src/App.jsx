import { useState, useRef, useCallback } from "react";

const API_ENDPOINT = "http://0.0.0.0:8000/recolor"; // 🔁 swap this

const PRESET_PALETTES = [
  { name: "Sunset", colors: ["#FF6B6B","#FF8E53","#FFC300","#FF5733","#C70039","#900C3F"] },
  { name: "Ocean",  colors: ["#0077B6","#00B4D8","#90E0EF","#CAF0F8","#023E8A","#03045E"] },
  { name: "Forest", colors: ["#2D6A4F","#40916C","#52B788","#74C69D","#95D5B2","#D8F3DC"] },
  { name: "Candy",  colors: ["#FF77AA","#FF99CC","#FFB347","#FFEC5C","#77DD77","#AEC6CF"] },
  { name: "Mono",   colors: ["#FFFFFF","#CCCCCC","#999999","#666666","#333333","#000000"] },
  { name: "Neon",   colors: ["#FF00FF","#00FFFF","#00FF00","#FF6600","#FF0066","#6600FF"] },
];

// Converts canvas pixel to hex
const rgbToHex = ([r,g,b]) => "#" + [r,g,b].map(v => v.toString(16).padStart(2,"0")).join("");
const hexToRgb = hex => { const c = hex.replace("#",""); return [0,2,4].map(i => parseInt(c.slice(i,i+2),16)); };

function ColorWheel({ onPick }) {
  const ref = useRef(null);
  const dragging = useRef(false);

  const draw = useCallback(canvas => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width: sz } = canvas;
    const cx = sz/2, cy = sz/2, r = sz/2 - 4;
    for (let a = 0; a < 360; a++) {
      const g = ctx.createRadialGradient(cx,cy,r*0.45,cx,cy,r);
      g.addColorStop(0,`hsla(${a},100%,50%,0)`);
      g.addColorStop(1,`hsl(${a},100%,50%)`);
      ctx.beginPath(); ctx.moveTo(cx,cy);
      ctx.arc(cx,cy,r,(a-1)*Math.PI/180,(a+1)*Math.PI/180);
      ctx.fillStyle = g; ctx.fill();
    }
    const wg = ctx.createRadialGradient(cx,cy,0,cx,cy,r*0.45);
    wg.addColorStop(0,"rgba(255,255,255,1)"); wg.addColorStop(1,"rgba(255,255,255,0)");
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fillStyle=wg; ctx.fill();
    const dg = ctx.createRadialGradient(cx,cy,r*0.45,cx,cy,r);
    dg.addColorStop(0,"rgba(0,0,0,0)"); dg.addColorStop(1,"rgba(0,0,0,0.28)");
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fillStyle=dg; ctx.fill();
  },[]);

  const pick = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.round((e.clientX-rect.left)*(canvas.width/rect.width));
    const y = Math.round((e.clientY-rect.top)*(canvas.height/rect.height));
    const px = canvas.getContext("2d").getImageData(x,y,1,1).data;
    onPick(rgbToHex([px[0],px[1],px[2]]));
  };

  return (
    <canvas ref={el=>{ref.current=el;draw(el);}} width={200} height={200}
      style={{borderRadius:"50%",cursor:"crosshair",border:"3px solid #ACBAC4",
        boxShadow:"0 0 32px rgba(225,217,188,0.3)",touchAction:"none"}}
      onMouseDown={e=>{dragging.current=true;pick(e,ref.current);}}
      onMouseMove={e=>{if(dragging.current)pick(e,ref.current);}}
      onMouseUp={()=>{dragging.current=false;}}
      onMouseLeave={()=>{dragging.current=false;}}
    />
  );
}

function Swatch({ color, index, selected, onClick }) {
  return (
    <button onClick={()=>onClick(index)} title={`Slot ${index+1}${color?": "+color:" (empty)"}`}
      style={{width:46,height:46,borderRadius:12,cursor:"pointer",position:"relative",
        backgroundColor:color||"#30364F",transition:"all 0.15s",
        border: selected?"3px solid #E1D9BC":"3px solid #ACBAC4",
        transform:selected?"scale(1.15)":"scale(1)",
        boxShadow:selected?"0 0 0 3px #E1D9BC":"none"}}>
      {!color && <span style={{position:"absolute",inset:0,display:"flex",alignItems:"center",
        justifyContent:"center",color:"#ACBAC4",fontSize:20}}>+</span>}
    </button>
  );
}

function DropZone({ onFile }) {
  const [drag,setDrag] = useState(false);
  const ref = useRef(null);
  const handle = f => { if(f?.type.startsWith("image/")) onFile(f); };
  return (
    <div onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)}
      onDrop={e=>{e.preventDefault();setDrag(false);handle(e.dataTransfer.files[0]);}}
      onClick={()=>ref.current.click()}
      style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",
        gap:10,borderRadius:16,height:170,cursor:"pointer",transition:"all 0.2s",
        border:`2px dashed ${drag?"#E1D9BC":"#ACBAC4"}`,
        background:drag?"rgba(225,217,188,0.15)":"transparent"}}>
      <span style={{fontSize:44,opacity:0.55}}>🖼️</span>
      <p style={{margin:0,color:"#E1D9BC",fontSize:13,fontWeight:500}}>Drop image or click to browse</p>
      <p style={{margin:0,color:"#ACBAC4",fontSize:11}}>PNG, JPG, WEBP</p>
      <input ref={ref} type="file" accept="image/*" style={{display:"none"}} onChange={e=>handle(e.target.files[0])}/>
    </div>
  );
}

const card = {background:"rgba(172,186,196,0.08)",borderRadius:20,padding:"20px 22px",border:"1px solid #ACBAC4"};
const lbl  = {fontSize:10,fontWeight:700,letterSpacing:"0.14em",color:"#ACBAC4",textTransform:"uppercase",marginBottom:14,display:"block"};

export default function ImageRecolor() {
  const [imgFile,setImgFile]   = useState(null);
  const [imgB64,setImgB64]     = useState(null);
  const [preview,setPreview]   = useState(null);
  const [palette,setPalette]   = useState(Array(6).fill(null));
  const [slot,setSlot]         = useState(0);
  const [result,setResult]     = useState(null);
  const [loading,setLoading]   = useState(false);
  const [error,setError]       = useState(null);
  const [tab,setTab]           = useState("wheel");
  const [hex,setHex]           = useState("#");

  const loadFile = file => {
    setImgFile(file); setResult(null); setError(null);
    const r = new FileReader();
    r.onload = e => { setPreview(e.target.result); setImgB64(e.target.result.split(",")[1]); };
    r.readAsDataURL(file);
  };

  const pickColor = color => {
    const next = [...palette]; next[slot] = color; setPalette(next);
    const ne = next.findIndex((c,i) => i > slot && !c);
    if (ne !== -1) setSlot(ne);
  };

  const filled = palette.filter(Boolean);

  const recolor = async () => {
    if (!imgB64)         return setError("Upload an image first.");
    if (!filled.length)  return setError("Pick at least one color.");
    setError(null); setLoading(true); setResult(null);
    try {
      const res = await fetch(API_ENDPOINT, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ image_base64: imgB64, palette: filled.map(hexToRgb) }),
      });
      if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResult(url);
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const T = (id,label) => (
    <button key={id} onClick={()=>setTab(id)} style={{flex:1,padding:"7px 0",borderRadius:10,fontSize:12,
      fontWeight:600,cursor:"pointer",border:"none",transition:"all 0.15s",
      background:tab===id?"#E1D9BC":"transparent",color:tab===id?"#30364F":"#ACBAC4"}}>
      {label}
    </button>
  );

  const disabled = loading || !imgB64 || !filled.length;

  return (
    <div style={{minHeight:"100vh",background:"#30364F",color:"#F0F0DB",
      fontFamily:"system-ui,sans-serif",padding:"28px 18px",
      display:"flex",flexDirection:"column",alignItems:"center"}}>

      <div style={{textAlign:"center",marginBottom:28}}>
        <h1 style={{fontSize:34,fontWeight:800,margin:0,letterSpacing:"-0.5px",
          background:"linear-gradient(135deg,#E1D9BC,#ACBAC4)",
          WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
          Recolorize!
        </h1>
        <p style={{margin:"6px 0 0",color:"#ACBAC4",fontSize:13}}>Upload · Pick palette · Transform</p>
      </div>

      <div style={{width:"100%",display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>

        {/* LEFT */}
        <div style={{display:"flex",flexDirection:"column",gap:18}}>

          {/* Upload */}
          <div style={card}>
            <span style={lbl}>1 · Upload Image</span>
            {preview ? (
              <div style={{position:"relative",borderRadius:12,overflow:"hidden",border:"1px solid #ACBAC4",display:"flex",justifyContent:"center",background:"rgba(0,0,0,0.2)"}}>
                <img src={preview} alt="original" style={{maxWidth:"100%",maxHeight:400,objectFit:"contain",display:"block"}}/>
                <button onClick={()=>{setImgFile(null);setImgB64(null);setPreview(null);setResult(null);}}
                  style={{position:"absolute",top:8,right:8,background:"#30364F",border:"none",
                    color:"#F0F0DB",borderRadius:"50%",width:26,height:26,cursor:"pointer",fontSize:13}}>✕</button>
                <div style={{position:"absolute",bottom:0,left:0,right:0,
                  background:"linear-gradient(transparent,rgba(0,0,0,0.6))",padding:"6px 12px 10px"}}>
                  <p style={{margin:0,fontSize:11,color:"#F0F0DB",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                    {imgFile?.name}
                  </p>
                </div>
              </div>
            ) : <DropZone onFile={loadFile}/>}
          </div>

          {/* Palette */}
          <div style={card}>
            <span style={lbl}>2 · Build Palette</span>

            <div style={{display:"flex",gap:8,marginBottom:18,justifyContent:"space-between"}}>
              {palette.map((c,i) => <Swatch key={i} color={c} index={i} selected={slot===i} onClick={setSlot}/>)}
            </div>

            <div style={{display:"flex",height:6,borderRadius:6,overflow:"hidden",
              marginBottom:18,border:"1px solid #ACBAC4"}}>
              {palette.map((c,i) => <div key={i} style={{flex:1,background:c||"rgba(172,186,196,0.15)"}}/>)}
            </div>

            <div style={{display:"flex",gap:4,background:"rgba(172,186,196,0.15)",
              borderRadius:12,padding:4,marginBottom:16}}>
              {T("wheel","🎨 Wheel")}
              {T("presets","✨ Presets")}
              {T("hex","# Hex")}
            </div>

            {tab==="wheel" && (
              <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:10}}>
                <p style={{margin:0,fontSize:12,color:"#ACBAC4"}}>
                  Slot <span style={{color:"#E1D9BC",fontWeight:700}}>{slot+1}</span>
                  {palette[slot] && <span style={{marginLeft:8,fontFamily:"monospace",color:palette[slot]}}>{palette[slot]}</span>}
                </p>
                <ColorWheel onPick={pickColor}/>
                <p style={{margin:0,fontSize:11,color:"#ACBAC4"}}>Click or drag to select</p>
              </div>
            )}

            {tab==="presets" && (
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
                {PRESET_PALETTES.map(p=>(
                  <button key={p.name} onClick={()=>{setPalette([...p.colors]);setSlot(0);}}
                    style={{display:"flex",alignItems:"center",gap:8,padding:"9px 10px",borderRadius:12,
                      background:"rgba(172,186,196,0.15)",border:"1px solid #ACBAC4",cursor:"pointer"}}>
                    <div style={{display:"flex",gap:2}}>
                      {p.colors.map((c,i)=>(
                        <div key={i} style={{width:11,height:20,backgroundColor:c,
                          borderRadius:i===0?"4px 0 0 4px":i===5?"0 4px 4px 0":0}}/>
                      ))}
                    </div>
                    <span style={{fontSize:12,color:"#F0F0DB",fontWeight:500}}>{p.name}</span>
                  </button>
                ))}
              </div>
            )}

            {tab==="hex" && (
              <div style={{display:"flex",flexDirection:"column",gap:10}}>
                <p style={{margin:0,fontSize:12,color:"#ACBAC4"}}>
                  Slot <span style={{color:"#E1D9BC",fontWeight:700}}>{slot+1}</span>
                </p>
                <div style={{display:"flex",gap:8}}>
                  <div style={{position:"relative",flex:1}}>
                    <input value={hex} onChange={e=>setHex(e.target.value)}
                      onKeyDown={e=>e.key==="Enter"&&/^#[0-9a-fA-F]{6}$/.test(hex)&&pickColor(hex)}
                      placeholder="#FF6B6B" maxLength={7}
                      style={{width:"100%",background:"rgba(172,186,196,0.15)",border:"1px solid #ACBAC4",
                        borderRadius:10,padding:"9px 36px 9px 12px",fontSize:13,fontFamily:"monospace",
                        color:"#F0F0DB",outline:"none",boxSizing:"border-box"}}/>
                    {/^#[0-9a-fA-F]{6}$/.test(hex) && (
                      <div style={{position:"absolute",right:10,top:"50%",transform:"translateY(-50%)",
                        width:18,height:18,borderRadius:"50%",backgroundColor:hex,
                        border:"1px solid #ACBAC4"}}/>
                    )}
                  </div>
                  <button onClick={()=>/^#[0-9a-fA-F]{6}$/.test(hex)&&pickColor(hex)}
                    style={{padding:"0 14px",background:"#E1D9BC",border:"none",borderRadius:10,
                      color:"#30364F",fontSize:13,fontWeight:600,cursor:"pointer"}}>Set</button>
                </div>
                <div style={{display:"flex",gap:7}}>
                  {["#FF6B6B","#FFD93D","#6BCB77","#4D96FF","#FF6BAE","#C77DFF"].map(c=>(
                    <button key={c} onClick={()=>{setHex(c);pickColor(c);}}
                      style={{width:28,height:28,borderRadius:8,backgroundColor:c,
                        border:"2px solid #ACBAC4",cursor:"pointer"}}/>
                  ))}
                </div>
              </div>
            )}

            <button onClick={()=>{setPalette(Array(6).fill(null));setSlot(0);}}
              style={{marginTop:14,width:"100%",background:"none",border:"none",
                color:"#ACBAC4",fontSize:11,cursor:"pointer"}}>
              ↺ Clear palette
            </button>
          </div>
        </div>

        {/* RIGHT */}
        <div style={{display:"flex",flexDirection:"column",gap:18}}>

          {/* CTA */}
          <div style={card}>
            <span style={lbl}>3 · Recolor</span>
            <div style={{display:"flex",height:28,borderRadius:10,overflow:"hidden",
              border:"1px solid #ACBAC4",marginBottom:10}}>
              {palette.map((c,i)=><div key={i} style={{flex:1,background:c||"rgba(172,186,196,0.15)"}}/>)}
            </div>
            <p style={{margin:"0 0 16px",fontSize:12,color:"#ACBAC4"}}>
              {filled.length} / 6 colors selected
            </p>
            <button onClick={recolor} disabled={disabled} style={{
              width:"100%",padding:"15px 0",borderRadius:16,border:"none",
              fontSize:16,fontWeight:700,cursor:disabled?"not-allowed":"pointer",transition:"all 0.2s",
              background:disabled?"rgba(172,186,196,0.15)":"linear-gradient(135deg,#E1D9BC,#ACBAC4)",
              color:disabled?"#ACBAC4":"#30364F",
              boxShadow:disabled?"none":"0 8px 28px rgba(225,217,188,0.4)"}}>
              {loading?"⏳ Processing…":"✦ Recolor Image"}
            </button>
            {error && (
              <div style={{marginTop:12,padding:"10px 14px",background:"rgba(239,68,68,0.15)",
                border:"1px solid #ef4444",borderRadius:10,fontSize:12,color:"#f87171"}}>
                ⚠ {error}
              </div>
            )}
          </div>

          {/* Result */}
          <div style={{...card,flex:1}}>
            <span style={lbl}>Result</span>
            {result ? (
              <div style={{display:"flex",flexDirection:"column",gap:12}}>
                <div style={{borderRadius:12,overflow:"hidden",border:"1px solid #ACBAC4",display:"flex",justifyContent:"center",background:"rgba(0,0,0,0.2)"}}>
                  <img src={result} alt="recolored" style={{maxWidth:"100%",maxHeight:400,display:"block",objectFit:"contain"}}/>
                </div>
                <a href={result} download="recolored.png" style={{
                  display:"block",textAlign:"center",padding:"10px 0",borderRadius:12,
                  background:"rgba(225,217,188,0.2)",border:"1px solid #E1D9BC",
                  color:"#E1D9BC",fontSize:13,fontWeight:600,textDecoration:"none"}}>
                  ↓ Download PNG
                </a>
              </div>
            ) : (
              <div style={{display:"flex",flexDirection:"column",alignItems:"center",
                justifyContent:"center",minHeight:180,gap:10,color:"#ACBAC4"}}>
                <span style={{fontSize:46}}>🎨</span>
                <p style={{margin:0,fontSize:13}}>Recolored image will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}