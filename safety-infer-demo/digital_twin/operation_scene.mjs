import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.mjs';

const SCALE=.24, colors=['#60a5fa','#34d399','#c4a4ff','#fb9678'];
export function createScene(container,onInspect){
  const scene=new THREE.Scene();scene.background=new THREE.Color('#10212d');
  const camera=new THREE.PerspectiveCamera(43,1,.05,250);camera.position.set(20,15,23);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);
  const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.set(0,6,0);orbit.enableDamping=true;orbit.maxPolarAngle=Math.PI*.49;
  scene.add(new THREE.HemisphereLight('#c8e7ef','#3e515c',2.3));const sun=new THREE.DirectionalLight('#fff5db',2.6);sun.position.set(8,30,15);scene.add(sun);
  const grid=new THREE.GridHelper(100,50,'#3e5967','#233e4f');scene.add(grid);
  const model=new THREE.Group(),paths=new THREE.Group(),obstacleGroup=new THREE.Group();scene.add(model,paths,obstacleGroup);
  const meshes=[];let current=null,previousData=null,cachedGeometry='',cachedPaths='',top,load,cable,mastSegments=[];
  function clear(group){group.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose()});group.clear()}
  function mat(){return new THREE.MeshStandardMaterial({color:'#73c9b2',roughness:.55,metalness:.22})}
  function box(group,size,position,id){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...size),mat());mesh.position.set(...position);mesh.userData.id=id;group.add(mesh);meshes.push(mesh);return mesh}
  function beam(group,a,b,r,id){const start=new THREE.Vector3(...a),end=new THREE.Vector3(...b);const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r,r,start.distanceTo(end),6),mat());mesh.position.copy(start).add(end).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),end.sub(start).normalize());mesh.userData.id=id;group.add(mesh);meshes.push(mesh);return mesh}
  function build(c){
    clear(model);clear(obstacleGroup);meshes.length=0;mastSegments=[];
    const h=c.equipment.tower_height_m*SCALE,jib=c.equipment.jib_length_m*SCALE,r=c.state.radius_m*SCALE;
    box(model,[1.6,.25,1.6],[0,.12,0],'base');
    for(let i=0;i<Math.ceil(h/1.2);i++){
      const count=Math.ceil(h/1.2),y0=h*i/count,y1=h*(i+1)/count,segment=new THREE.Group();model.add(segment);mastSegments.push({group:segment,height:(y0+y1)/2});
      for(const x of [-.25,.25])for(const z of [-.25,.25])beam(segment,[x,y0,z],[x,y1,z],.045,'mast');
      for(const z of [-.25,.25]){beam(segment,[-.25,y0,z],[.25,y1,z],.022,'mast');beam(segment,[-.25,y1,z],[.25,y1,z],.025,'mast')}
    }
    c.equipment.attachment_heights_m.forEach(y=>{beam(model,[-.25,y*SCALE,-.2],[-2.5,y*SCALE,-1],.04,'attachment');beam(model,[-.25,y*SCALE,.2],[-2.5,y*SCALE,1],.04,'attachment')});
    top=new THREE.Group();model.add(top);top.position.y=h;
    box(top,[.7,.35,.7],[0,.1,0],'mast');
    for(const z of [-.22,.22]){beam(top,[0,0,z],[jib,0,z],.06,'jib');beam(top,[0,0,z],[-3,0,z],.055,'jib')}
    for(let x=0;x<jib;x+=1.2){beam(top,[x,0,-.22],[Math.min(jib,x+1.2),0,.22],.022,'jib');beam(top,[x,0,-.22],[x,0,.22],.024,'jib')}
    beam(top,[0,0,0],[0,2.4,0],.07,'mast');beam(top,[0,2.4,0],[jib*.75,0,0],.035,'jib');beam(top,[0,2.4,0],[-2.6,0,0],.035,'jib');
    box(top,[.7,.6,.7],[-2,-.3,0],'jib');box(top,[.4,.2,.4],[r,-.1,0],'hoist');
    load=box(model,[c.equipment.payload_radius_m*2*SCALE,.55,c.equipment.payload_radius_m*2*SCALE],[r,h-c.state.rope_length_m*SCALE,0],'hoist');
    cable=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:'#d9e3e5'}));model.add(cable);
    if(c.obstacle.enabled){const o=c.obstacle;const mesh=new THREE.Mesh(new THREE.CylinderGeometry(o.radius_m*SCALE,o.radius_m*SCALE,h*.75,32),new THREE.MeshStandardMaterial({color:'#9d785d',transparent:true,opacity:.28,depthWrite:false}));mesh.position.set(o.x_m*SCALE,h*.375,o.z_m*SCALE);obstacleGroup.add(mesh);const ring=new THREE.Mesh(new THREE.RingGeometry(o.radius_m*SCALE,(o.radius_m+c.limits.min_clearance_m+c.equipment.payload_radius_m)*SCALE,48),new THREE.MeshBasicMaterial({color:'#d4a779',transparent:true,opacity:.14,side:THREE.DoubleSide}));ring.rotation.x=-Math.PI/2;ring.position.set(o.x_m*SCALE,.01,o.z_m*SCALE);obstacleGroup.add(ring)}
    camera.position.set(h*1.65,h*1.25,h*1.9);orbit.target.set(0,h*.55,0);orbit.update();
  }
  function update(data,selected,index,showAll,amplify){
    if(previousData!==data){cachedPaths='';previousData=data}
    current={data,selected,index};const c=data.config,p=data.scenarios[selected].series[index];
    const geometryKey=JSON.stringify([c.state.radius_m,c.state.rope_length_m,c.equipment,c.obstacle]);
    if(cachedGeometry!==geometryKey){build(c);cachedGeometry=geometryKey}
    const multiplier=amplify?20:1,qx=p.mast_x_mm.mean/1000*SCALE,qz=p.mast_z_mm.mean/1000*SCALE,h=c.equipment.tower_height_m*SCALE;
    const freeBottom=(c.equipment.attachment_heights_m.at(-1)||0)*SCALE;
    for(const segment of mastSegments){const f=Math.max(0,(segment.height-freeBottom)/(h-freeBottom));segment.group.position.set(qx*multiplier*f*f,0,qz*multiplier*f*f)}
    top.position.set(qx*multiplier,h,qz*multiplier);top.rotation.y=-p.slew_deg*Math.PI/180;
    load.position.set(p.load_x_m.mean*SCALE+(multiplier-1)*qx,p.load_y_m.mean*SCALE,p.load_z_m.mean*SCALE+(multiplier-1)*qz);
    const phi=p.slew_deg*Math.PI/180,r=c.state.radius_m*SCALE,pivot=new THREE.Vector3(r*Math.cos(phi)+qx*multiplier,h,r*Math.sin(phi)+qz*multiplier);
    cable.geometry.setFromPoints([pivot,load.position]);
    const utilization={mast:p.mast_moment_kn_m.mean/c.limits.mast_moment_kn_m,base:p.mast_moment_kn_m.mean/c.limits.mast_moment_kn_m,attachment:p.attachment_axial_kn.mean/c.limits.attachment_axial_kn,hoist:c.state.load_t/c.equipment.rated_load_t,jib:c.state.load_t*9.81*c.state.radius_m/c.equipment.rated_lift_moment_kn_m};
    meshes.forEach(mesh=>mesh.material.color.set(utilization[mesh.userData.id]>=1?'#ef8d7e':utilization[mesh.userData.id]>=.7?'#d9b578':'#6bbbaa'));
    const pathKey=data.ensemble.id+':'+showAll+':'+selected;
    if(cachedPaths!==pathKey){
      clear(paths);data.scenarios.forEach((s,i)=>{if(!showAll&&i!==selected)return;const y=c.equipment.tower_height_m-c.state.rope_length_m;
        for(const path of s.trajectory_samples_xz_m.slice(0,12)){const points=path.map(p=>new THREE.Vector3(p[0]*SCALE,y*SCALE,p[1]*SCALE));paths.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:colors[i],transparent:true,opacity:.15})))}
        paths.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(s.series.map(p=>new THREE.Vector3(p.load_x_m.mean*SCALE,p.load_y_m.mean*SCALE,p.load_z_m.mean*SCALE))),new THREE.LineBasicMaterial({color:colors[i],transparent:true,opacity:.85})));
      });cachedPaths=pathKey;
    }
  }
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('click',e=>{if(!current)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(meshes).find(x=>x.object.userData.id);if(!hit)return;const {data,selected,index}=current,p=data.scenarios[selected].series[index],id=hit.object.userData.id;
    const info={mast:`塔身 · 弯矩 ${p.mast_moment_kn_m.mean.toFixed(1)} kN·m · X/Z 位移 ${p.mast_x_mm.mean.toFixed(1)} / ${p.mast_z_mm.mean.toFixed(1)} mm`,base:`基础示意 · 等效塔身弯矩 ${p.mast_moment_kn_m.mean.toFixed(1)} kN·m`,attachment:`附着 · 等效轴力 ${p.attachment_axial_kn.mean.toFixed(1)} kN（平均分配）`,jib:`起重臂 · 静态吊重力矩 ${(data.config.state.load_t*9.81*data.config.state.radius_m).toFixed(1)} kN·m`,hoist:`吊物 · 摆角 ${p.swing_deg.mean.toFixed(2)}° · X/Z ${p.load_x_m.mean.toFixed(2)} / ${p.load_z_m.mean.toFixed(2)} m`};onInspect(info[id]);
  });
  new ResizeObserver(()=>{const rect=container.getBoundingClientRect();if(rect.width&&rect.height){renderer.setSize(rect.width,rect.height);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}}).observe(container);
  function animate(){requestAnimationFrame(animate);orbit.update();renderer.render(scene,camera)}animate();
  return {update};
}
