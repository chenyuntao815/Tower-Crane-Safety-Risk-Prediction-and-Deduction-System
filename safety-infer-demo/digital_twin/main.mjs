import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.mjs';

const $ = id => document.getElementById(id);
const scalarFields = ['load_t','radius_m','wind_m_s','slew_deg','tower_height_m','jib_length_m','mast_ei_kn_m2','rated_moment_kn_m','mast_moment_capacity_kn_m','attachment_capacity_kn','attachment_lever_arm_m','rated_load_t','max_free_height_m','projected_area_m2','drag_coefficient'];
const unitScale = 0.24;
let data = null, selectedId = null, currentIndex = 0, config = null;
let playTimer = null;
let modelGroup = new THREE.Group(), ghostGroup = new THREE.Group();
const scene = new THREE.Scene();
scene.background = new THREE.Color('#10232d');
scene.fog = new THREE.Fog('#10232d', 25, 65);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 150);
camera.position.set(18, 14, 24);
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.65;
$('scene').appendChild(renderer.domElement);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 6, 0);
orbit.enableDamping = true;
orbit.maxPolarAngle = Math.PI * 0.48;
orbit.minDistance = 12;
orbit.maxDistance = 65;
scene.add(new THREE.HemisphereLight('#c5e8f2','#47616b',2.1));
const sun = new THREE.DirectionalLight('#fff0d5',2.5); sun.position.set(12,22,10); scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(100,100),new THREE.MeshStandardMaterial({color:'#122934',roughness:1}));
ground.rotation.x = -Math.PI/2; ground.position.y = -0.06; scene.add(ground);
const grid = new THREE.GridHelper(80,40,'#284650','#213d47'); grid.position.y = -0.045; scene.add(grid);
const building = new THREE.Mesh(new THREE.BoxGeometry(4.3,11,6),new THREE.MeshStandardMaterial({color:'#29424d',transparent:true,opacity:.46,roughness:1}));
building.position.set(-4.2,5.5,0);scene.add(building);
scene.add(modelGroup,ghostGroup);
const raycaster = new THREE.Raycaster(); const pointer = new THREE.Vector2();

function material(util,ghost=false){
  const color = ghost ? '#89d9ff' : util >= 1 ? '#f3766d' : util >= .7 ? '#efbc6c' : '#4bc8c1';
  return new THREE.MeshStandardMaterial({color,metalness:ghost?0:.35,roughness:.5,transparent:ghost,opacity:ghost?.28:1,depthWrite:!ghost});
}
function addBox(parent,w,h,d,x,y,z,cid,util,ghost=false){
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),material(util,ghost));
  mesh.position.set(x,y,z); mesh.userData.cid=cid; parent.add(mesh); return mesh;
}
function beam(parent,a,b,r,cid,util,ghost=false){
  const start=new THREE.Vector3(...a),end=new THREE.Vector3(...b);const length=start.distanceTo(end);
  const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r,r,length,6),material(util,ghost));
  mesh.position.copy(start).add(end).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),end.sub(start).normalize());
  mesh.userData.cid=cid;parent.add(mesh);
}
function utilFor(snapshot,id){return snapshot.components.find(c=>c.id===id)?.utilization||0}
function createCrane(snapshot,params,ghost=false){
  const group=ghost?ghostGroup:modelGroup;
  group.traverse(object=>{if(object.isMesh){object.geometry.dispose();object.material.dispose()}});
  group.clear();
  const h=snapshot.tower_height_m*unitScale, jib=params.jib_length_m*unitScale, radius=params.radius_m*unitScale;
  const mastUtil=utilFor(snapshot,'TC-MAST'),jibUtil=utilFor(snapshot,'TC-JIB'),loadUtil=utilFor(snapshot,'TC-HOIST');
  const width=.54;
  if(!ghost)addBox(group,2,.35,2,0,.1,0,'TC-BASE',utilFor(snapshot,'TC-BASE'));
  const rings=Math.max(4,Math.ceil(h/1.3));
  for(let i=0;i<rings;i++){
    const y0=h*i/rings,y1=h*(i+1)/rings;
    for(const x of [-width/2,width/2])for(const z of [-width/2,width/2])beam(group,[x,y0,z],[x,y1,z],.055,'TC-MAST',mastUtil,ghost);
    if(i%2===0){for(const z of [-width/2,width/2])beam(group,[-width/2,y0,z],[width/2,y1,z],.025,'TC-MAST',mastUtil,ghost);}
  }
  if(!ghost){
    snapshot.attachments_m.forEach((elevation,i)=>{
      const y=elevation*unitScale, id=`TC-ATT-${i+1}`,u=utilFor(snapshot,id);
      beam(group,[-.27,y,0],[-2.3,y,0],.07,id,u);
      beam(group,[-.27,y,.25],[-2.3,y,1.1],.04,id,u);
      addBox(group,.45,.34,.45,-2.3,y,0,id,u);
    });
  }
  const top=new THREE.Group();top.position.set(snapshot.displacement_mm/1000*unitScale*12,h,0);top.rotation.y=-snapshot.slew_deg*Math.PI/180;group.add(top);
  addBox(top,.8,.42,.8,0,.15,0,'TC-MAST',mastUtil,ghost);
  beam(top,[0,.12,-.28],[jib,.12,-.28],.07,'TC-JIB',jibUtil,ghost);
  beam(top,[0,.12,.28],[jib,.12,.28],.07,'TC-JIB',jibUtil,ghost);
  for(let x=0;x<jib;x+=1.5){beam(top,[x,.12,-.28],[Math.min(jib,x+1.5),.12,.28],.025,'TC-JIB',jibUtil,ghost);}
  beam(top,[0,.1,0],[-3.1,.1,0],.1,'TC-JIB',jibUtil,ghost);
  beam(top,[0,3.1,0],[jib*.78,.12,0],.045,'TC-JIB',jibUtil,ghost);
  beam(top,[0,3.1,0],[-3.1,.12,0],.045,'TC-JIB',jibUtil,ghost);
  beam(top,[0,.3,0],[0,3.1,0],.08,'TC-MAST',mastUtil,ghost);
  addBox(top,.48,.28,.6,radius,.04,0,'TC-HOIST',loadUtil,ghost);
  const cable=2.3;beam(top,[radius,0,0],[radius,-cable,0],.018,'TC-HOIST',loadUtil,ghost);
  addBox(top,.85,.8,.8,radius,-cable-.45,0,'TC-HOIST',loadUtil,ghost);
  if(!ghost)addBox(top,1,.9,.85,-1.65,-.35,0,'TC-JIB',jibUtil);
}

function resize(){const r=$('scene').getBoundingClientRect();renderer.setSize(r.width,r.height);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()}
new ResizeObserver(resize).observe($('scene'));resize();
function animate(){requestAnimationFrame(animate);orbit.update();renderer.render(scene,camera)}animate();
renderer.domElement.addEventListener('click',e=>{
  const bounds=renderer.domElement.getBoundingClientRect();
  pointer.set((e.clientX-bounds.left)/bounds.width*2-1,-(e.clientY-bounds.top)/bounds.height*2+1);
  raycaster.setFromCamera(pointer,camera);
  const hit=raycaster.intersectObjects(modelGroup.children,true).find(item=>item.object.userData.cid);
  if(hit){selectedId=hit.object.userData.cid;updateInspector()}
});

function esc(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function sample(){return data.scenarios[0].series[currentIndex]}
function updateInspector(){
  if(!data)return;const c=sample().components.find(item=>item.id===selectedId);
  $('inspector').innerHTML=c?`<div class="inspector-kicker">${esc(c.id)}</div><strong>${esc(c.name)} · 利用率 ${(c.utilization*100).toFixed(1)}%</strong><span>作用力 ${c.force_kn.toFixed(1)} kN ｜ 弯矩 ${c.moment_kn_m.toFixed(1)} kN·m${c.elevation_m!==undefined?` ｜ 标高 ${c.elevation_m} m`:''}</span>`:
    '<div class="inspector-kicker">构件受力</div><strong>点击塔吊构件查看详情</strong><span>颜色只对应结构利用率</span>';
}
function drawChart(id,values,color,index){
  const svg=$(id);const max=Math.max(...values,0.01)*1.12,min=0;
  const pts=values.map((v,i)=>`${10+i*280/(values.length-1)},${65-(v-min)/(max-min)*52}`).join(' ');
  const markerX=10+index*280/(values.length-1),markerY=65-(values[index]-min)/(max-min)*52;
  svg.innerHTML=`<line x1="10" x2="290" y1="65" y2="65" stroke="#38535c"/><line x1="10" x2="290" y1="35" y2="35" stroke="#263f49" stroke-dasharray="3 4"/><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/><circle cx="${markerX}" cy="${markerY}" r="4" fill="${color}" stroke="#132934" stroke-width="2"/>`;
}
function refresh(){
  if(!data)return;
  const s=sample(),future=data.scenarios[0].future;
  $('timeLabel').textContent=`${s.time_s} / ${data.horizon_s} s`;
  $('mDisp').textContent=s.displacement_mm.toFixed(1);
  $('mMoment').textContent=s.mast_moment_kn_m.toFixed(0);
  $('mAxial').textContent=s.attachment_axial_kn.toFixed(1);
  $('mRisk').textContent=(s.risk_probability*100).toFixed(1)+'%';
  const series=data.scenarios[0].series;
  drawChart('chartDisp',series.map(x=>x.displacement_mm),'#63c8d0',currentIndex);
  drawChart('chartMoment',series.map(x=>x.mast_moment_kn_m),'#dda75b',currentIndex);
  drawChart('chartAxial',series.map(x=>x.attachment_axial_kn),'#92bbec',currentIndex);
  drawChart('chartRisk',series.map(x=>x.risk_probability*100),'#ee8d78',currentIndex);
  $('riskList').innerHTML=s.risks.slice(0,4).map(r=>`<div class="risk-row"><span>${esc(r.risk_name)} <small>${esc(r.risk_id)}</small></span><b>${(r.prob*100).toFixed(1)}%</b></div>`).join('');
  createCrane(s,config);createCrane(future,config,true);updateInspector();
}
function showScenarios(){
  const baseline=data.scenarios[0].future;
  $('scenarioCards').innerHTML=data.scenarios.map(s=>{
    const f=s.future,delta=(f.risk_probability-baseline.risk_probability)*100;
    return `<article class="scenario-card"><h3>${esc(s.label)}<span class="tag">${s.load_t} t · ${s.radius_m} m</span></h3><div class="numbers"><div><small>位移</small><strong>${f.displacement_mm.toFixed(1)} <small>mm</small></strong></div><div><small>最大利用率</small><strong>${(f.max_utilization*100).toFixed(0)}%</strong></div><div><small>情景概率</small><strong>${(f.risk_probability*100).toFixed(1)}%</strong></div></div><p class="delta">相对原方案风险 ${delta<=0?'↓':'↑'} ${Math.abs(delta).toFixed(1)} 个百分点</p></article>`
  }).join('');
}
function readConfig(){
  const result={event:$('event').value};for(const key of scalarFields){const v=Number($(key).value);if(!Number.isFinite(v)||$(key).value.trim()==='')throw Error(`请填写有效的${$(key).previousElementSibling.textContent.trim()}`);result[key]=v}
  result.attachment_heights_m=$('attachment_heights_m').value.split(',').map(x=>Number(x.trim())).filter(x=>Number.isFinite(x)&&x>0);
  if(result.attachment_heights_m.length!==$('attachment_heights_m').value.split(',').length)throw Error('请用逗号分隔有效的附着高度');
  return result;
}
async function run(){
  $('error').hidden=true;
  try{
    if(playTimer){clearInterval(playTimer);playTimer=null;$('playButton').textContent='▶ 播放'}
    config=readConfig();const response=await fetch('/api/v1/twin/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
    const body=await response.json();if(!response.ok||body.code!==0)throw Error(typeof body.detail==='string'?body.detail:body.message||'推演失败');
    data=body.data;currentIndex=0;$('timeSlider').value=0;selectedId=null;
    $('graphSource').textContent=data.graph_source;
    $('mappingList').innerHTML=data.indicator_mappings.map(m=>`<div class="mapping-row"><span>${esc(m.indicator_id)} · ${esc(m.name)}</span><b>${m.value} ${esc(m.unit)}</b></div>`).join('');
    showScenarios();refresh();$('adviceContent').textContent='推演已更新。点击“生成 AI 建议”调用 DeepSeek。';
  }catch(error){$('error').textContent=error.message;$('error').hidden=false;$('graphSource').textContent='推演接口未连接'}
}
$('controls').addEventListener('submit',event=>{event.preventDefault();run()});
$('timeSlider').addEventListener('input',event=>{currentIndex=Number(event.target.value);refresh()});
$('playButton').addEventListener('click',()=>{
  if(!data)return;
  if(playTimer){clearInterval(playTimer);playTimer=null;$('playButton').textContent='▶ 播放';return}
  if(currentIndex>=12)currentIndex=0;
  $('playButton').textContent='Ⅱ 暂停';
  playTimer=setInterval(()=>{
    currentIndex+=1;$('timeSlider').value=currentIndex;refresh();
    if(currentIndex>=12){clearInterval(playTimer);playTimer=null;$('playButton').textContent='▶ 播放'}
  },600);
});
$('adviceButton').addEventListener('click',async()=>{
  if(!data){$('adviceContent').textContent='请先运行推演。';return}
  const button=$('adviceButton');button.disabled=true;$('adviceContent').textContent='正在请求 DeepSeek…';
  try{
    const twin_scenarios={event:data.event_name,model_status:data.model_status,graph_source:data.graph_source,
      inputs:{load_t:config.load_t,radius_m:config.radius_m,wind_m_s:config.wind_m_s,slew_deg:config.slew_deg},
      measures:data.scenarios.map(s=>({name:s.label,load_t:s.load_t,radius_m:s.radius_m,
        displacement_mm:s.future.displacement_mm,moment_kn_m:s.future.mast_moment_kn_m,
        attachment_axial_kn:s.future.attachment_axial_kn,risk_probability:s.future.risk_probability,
        top_risks:s.future.risks.slice(0,3)}))};
    const response=await fetch('/api/v1/infer/llm-decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({infer_result:{twin_scenarios}})});
    const body=await response.json();if(!response.ok||body.code!==0)throw Error(body.message||'AI 请求失败');
    const d=body.data;
    $('adviceContent').textContent=[d.situation_analysis,...d.main_warnings.map(x=>'警示：'+x),...d.prevention_measures.map(x=>'措施：'+x),'应急：'+d.emergency_response,'监测：'+d.monitoring_focus].join('\n');
  }catch(error){$('adviceContent').textContent='AI 建议生成失败：'+error.message}
  finally{button.disabled=false}
});
try {
  // Draw the crane immediately. A slow or unavailable graph API must not
  // leave the structure viewport empty.
  config=readConfig();
  createCrane({tower_height_m:config.tower_height_m,
    attachments_m:config.attachment_heights_m,
    displacement_mm:0,slew_deg:config.slew_deg,components:[]},config);
  window.__twinReady=true;
  $('sceneStatus').hidden=true;
  run();
} catch(error) {
  $('sceneStatus').textContent='三维模型初始化失败：'+error.message;
}
