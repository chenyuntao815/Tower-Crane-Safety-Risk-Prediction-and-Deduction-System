// Regression tests execute the real frontend against a detached DOM and mocked API.
// Run: node --experimental-vm-modules --test test_operation_inputs.mjs
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const html=await readFile(new URL('./digital_twin/operations.html',import.meta.url),'utf8');
const source=await readFile(new URL('./digital_twin/operations.mjs',import.meta.url),'utf8');
const defaults=JSON.parse(await readFile(new URL('./examples/obstacle-input.json',import.meta.url),'utf8'));
const labels={manual:'手动输入（当前数据）',default:'示例：常规回转',obstacle:'示例：前方障碍物',swing:'示例：已有摆动',gust:'示例：强阵风',overload:'示例：超额定工况'};

function responseFor(config){
  const band=v=>({mean:v,p05:v-.1,p95:v+.1});
  const events=Object.fromEntries(['swing','moment','attachment','clearance'].map(key=>[key,{probability:.1}]));
  const scenarios=['maintain','early_deceleration','smooth_brake','immediate_brake'].map((id,i)=>({
    id,label:Object.values(labels)[i+1],risk:{probability:.1,ci95:[0,.2],events},
    comparison:{risk_reduction_pp:0,paired_delta_ci95_pp:[-1,1]},
    peaks:{swing_deg:band(1),mast_moment_kn_m:band(100),attachment_axial_kn:band(10)},
    costs:{estimated_completion_s:10,stop_cost:0,risk_cost:100,time_cost:10,total:110,completion_within_horizon_s:null,remaining_slew_deg:5},
    constraints:{feasible:true,reasons:[],peak_acceleration_deg_s2:1,peak_jerk_deg_s3:1},validity:{small_angle_valid:true},
    knowledge_rule:{applies_when:['test'],tradeoff:'test'},trajectory_samples_xz_m:[[[10,20],[11,21]]],
    series:[0,config.horizon_s].map(time_s=>({time_s,swing_deg:band(1),mast_moment_kn_m:band(100),attachment_axial_kn:band(10),
      load_x_m:band(10),load_z_m:band(20),mast_x_mm:band(5),mast_z_mm:band(5),
      cumulative_probability:.1,instantaneous_probability:.1,event_probabilities:{swing:.1,moment:.1,attachment:.1,clearance:.1}}))
  }));
  return {config,scenarios,recommendation:{action_id:'maintain',status:'within_budget',text:'test'},ensemble:{id:'test'},
    graph_source:'test',measure_knowledge_source:'test',knowledge_chains:[],warnings:[],uncertainty_note:'test',
    event_names:{swing:'摆角',moment:'弯矩',attachment:'轴力',clearance:'距离'},
    estimation:{swing_state:'test',mast_posterior_std_mm:[1,1],parameter_status:'test',wind_process:'test'}};
}

async function frontend(){
  const elements=new Map(),requests=[],adviceRequests=[],routes=[];
  let nextSimulation,nextAdvice;
  const explanation={situation_analysis:'根据计算结果解释当前工况',main_warnings:['留意摆动'],prevention_measures:['平缓操作'],emergency_response:'按预案处置',monitoring_focus:'持续监测'};
  const adviceFor=()=>({explanation,ensemble_id:'test'});
  const decode=value=>value.replaceAll('&quot;','"').replaceAll('&#39;',"'").replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&');
  class Element{
    constructor(id){this.id=id;this.value='';this.checked=false;this.hidden=false;this.disabled=false;this.listeners={};this.classList={toggle(){}}}
    addEventListener(type,handler){(this.listeners[type]??=[]).push(handler)}
    async dispatch(type,event={}){for(const handler of this.listeners[type]??[])await handler({target:this,preventDefault(){},...event})}
    set innerHTML(value){this.content=value;for(const match of value.matchAll(/<input\b([^>]+)>/g)){
      const attrs=match[1],id=attrs.match(/\bid="([^"]+)"/)?.[1];if(!id)continue;
      const input=new Element(id);input.value=decode(attrs.match(/\bvalue="([^"]*)"/)?.[1]??'');input.checked=/\bchecked\b/.test(attrs);elements.set(id,input);
    }}
    get innerHTML(){return this.content??''}
    get selectedOptions(){return [{textContent:labels[this.value]}]}
    querySelector(){return new Element('tbody')}
    insertAdjacentHTML(_position,value){this.content+=value}
  }
  for(const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)){
    const element=new Element(match[1]);element.disabled=/\bdisabled\b/.test(match[0]);elements.set(match[1],element);
  }
  elements.get('preset').value=html.match(/<select id="preset"><option value="([^"]+)"/)?.[1];
  const context=vm.createContext({structuredClone,console,performance,setInterval,clearInterval,setTimeout,URL,Blob,
    document:{getElementById:id=>elements.get(id),querySelectorAll:()=>[]},
    fetch:async (url,options)=>{
      routes.push(url.split('/').at(-1));
      if(url.endsWith('/defaults'))return {ok:true,json:async()=>({code:0,data:structuredClone(defaults)})};
      const config=JSON.parse(options.body);let data;
      if(url.endsWith('/advice')){
        adviceRequests.push(config);const handler=nextAdvice;nextAdvice=undefined;
        data=handler?await handler(config):adviceFor();
      }else{
        requests.push(config);const handler=nextSimulation;nextSimulation=undefined;
        data=handler?await handler(config):responseFor(config);
      }
      return {ok:true,json:async()=>({code:0,data})};
    }});
  const module=new vm.SourceTextModule(source,{context,importModuleDynamically:async()=>{
    const renderer=new vm.SyntheticModule(['createScene'],function(){this.setExport('createScene',()=>({update(){}}))},{context});
    await renderer.link(()=>{});await renderer.evaluate();return renderer;
  }});
  await module.link(()=>{});await module.evaluate();
  const get=id=>elements.get(id);
  return {get,requests,adviceRequests,routes,
    async edit(id,value){get(id).value=String(value);await get('operationForm').dispatch('input')},
    async preset(value){get('preset').value=value;await get('preset').dispatch('change')},
    defer(){let resolve;const pending=new Promise(r=>{resolve=r});nextSimulation=async config=>{await pending;return responseFor(config)};return resolve},
    deferAdvice(){let resolve;const pending=new Promise(r=>{resolve=r});nextAdvice=async()=>{await pending;return adviceFor()};return resolve},
    failAdvice(){nextAdvice=async()=>{throw Error('DeepSeek 认证失败，请检查后端密钥配置')}},
    failSimulation(){nextSimulation=async()=>{throw Error('输入参数超出允许范围')}}
  };
}

test('manual load preserves edited state, equipment, uncertainty and task parameters',async()=>{
  const ui=await frontend();assert.equal(ui.get('preset').value,'manual');
  const edits={state_load_t:7.3,state_wind_m_s:10.2,state_rope_length_m:22,state_swing_x_deg:2.1,
    equipment_mast_ei_kn_m2:3000000,uncertainty_gust_std_m_s:2.2,controls_stop_cost:60,run_samples:64};
  for(const [id,value] of Object.entries(edits))await ui.edit(id,value);
  await ui.get('loadPreset').dispatch('click');
  const sent=ui.requests.at(-1);
  assert.equal(sent.state.load_t,7.3);assert.equal(sent.state.wind_m_s,10.2);assert.equal(sent.state.rope_length_m,22);
  assert.equal(sent.state.swing_x_deg,2.1);assert.equal(sent.equipment.mast_ei_kn_m2,3000000);
  assert.equal(sent.uncertainty.gust_std_m_s,2.2);assert.equal(sent.controls.stop_cost,60);assert.equal(sent.samples,64);
  for(const [id,value] of Object.entries(edits))assert.equal(ui.get(id).value,String(value));
  assert.equal(ui.get('error').hidden,true);
});

test('example loading only fills the form, then further loads use edited data',async()=>{
  const ui=await frontend();await ui.preset('gust');
  assert.equal(ui.get('loadPreset').textContent,'填入所选示例');
  const count=ui.requests.length;await ui.get('loadPreset').dispatch('click');
  assert.equal(ui.requests.length,count);assert.equal(ui.get('state_wind_m_s').value,'18');
  assert.equal(ui.get('preset').value,'manual');
  await ui.edit('state_wind_m_s',11.8);await ui.get('loadPreset').dispatch('click');
  assert.equal(ui.requests.at(-1).state.wind_m_s,11.8);assert.equal(ui.get('state_wind_m_s').value,'11.8');
});

test('editing after selecting an example switches back to manual input',async()=>{
  const ui=await frontend();await ui.preset('gust');await ui.edit('state_wind_m_s',9.4);
  assert.equal(ui.get('preset').value,'manual');await ui.get('loadPreset').dispatch('click');
  assert.equal(ui.requests.at(-1).state.wind_m_s,9.4);
});

test('invalid manual values remain visible and can be corrected',async()=>{
  const ui=await frontend(),count=ui.requests.length;await ui.edit('state_wind_m_s','');
  await ui.get('loadPreset').dispatch('click');assert.equal(ui.requests.length,count);
  assert.equal(ui.get('state_wind_m_s').value,'');assert.equal(ui.get('error').hidden,false);
  await ui.edit('state_wind_m_s',12);await ui.get('operationForm').dispatch('submit');
  // A DOM submit callback starts the request without returning its Promise.
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(ui.requests.at(-1).state.wind_m_s,12);assert.equal(ui.get('runButton').disabled,false);
});

test('in-flight responses do not overwrite edits or leave the run button locked',async()=>{
  const ui=await frontend(),release=ui.defer();await ui.edit('state_wind_m_s',10);
  const pending=ui.get('loadPreset').dispatch('click');assert.equal(ui.get('runButton').disabled,true);
  await ui.edit('state_wind_m_s','');await ui.get('loadPreset').dispatch('click');release();await pending;
  assert.equal(ui.get('state_wind_m_s').value,'');assert.equal(ui.get('runButton').disabled,false);
  assert.match(ui.get('runStatus').textContent,/参数已修改/);
});

test('imported parameters enter manual mode even if an example was selected',async()=>{
  const ui=await frontend(),config=structuredClone(defaults);config.state.load_t=8.2;
  await ui.preset('gust');ui.get('importConfig').files=[{size:100,text:async()=>JSON.stringify(config)}];
  await ui.get('importConfig').dispatch('change');assert.equal(ui.get('preset').value,'manual');
  await ui.get('loadPreset').dispatch('click');assert.equal(ui.requests.at(-1).state.load_t,8.2);
});

test('AI button remains clickable after edits and refreshes simulation before explaining',async()=>{
  const ui=await frontend();assert.equal(ui.get('adviceButton').disabled,false);
  await ui.edit('state_wind_m_s',13.7);assert.equal(ui.get('adviceButton').disabled,false);
  await ui.get('adviceButton').dispatch('click');
  assert.deepEqual(ui.routes.slice(-2),['simulate','advice']);
  assert.equal(ui.requests.at(-1).state.wind_m_s,13.7);
  assert.deepEqual(ui.adviceRequests.at(-1),ui.requests.at(-1));
  assert.match(ui.get('adviceContent').textContent,/根据计算结果/);
  assert.equal(ui.get('adviceButton').disabled,false);
});

test('AI can explain an explicitly loaded example without a separate run click',async()=>{
  const ui=await frontend();await ui.preset('gust');await ui.get('loadPreset').dispatch('click');
  assert.equal(ui.get('adviceButton').disabled,false);await ui.get('adviceButton').dispatch('click');
  assert.equal(ui.adviceRequests.at(-1).state.wind_m_s,18);
  assert.deepEqual(ui.routes.slice(-2),['simulate','advice']);
});

test('unchanged results are explained directly, and provider errors allow retry',async()=>{
  const ui=await frontend(),count=ui.requests.length;ui.failAdvice();
  await ui.get('adviceButton').dispatch('click');assert.equal(ui.requests.length,count);
  assert.match(ui.get('adviceContent').textContent,/认证失败/);assert.equal(ui.get('adviceButton').disabled,false);
  await ui.get('adviceButton').dispatch('click');assert.match(ui.get('adviceContent').textContent,/根据计算结果/);
});

test('invalid input and failed simulations never send an AI request or lock its button',async()=>{
  const ui=await frontend();await ui.edit('state_wind_m_s','');
  await ui.get('adviceButton').dispatch('click');assert.equal(ui.adviceRequests.length,0);
  assert.equal(ui.get('error').hidden,false);assert.equal(ui.get('adviceButton').disabled,false);
  await ui.edit('state_wind_m_s',12);ui.failSimulation();await ui.get('adviceButton').dispatch('click');
  assert.equal(ui.adviceRequests.length,0);assert.match(ui.get('error').textContent,/超出允许范围/);
  assert.equal(ui.get('adviceButton').disabled,false);assert.equal(ui.get('runButton').disabled,false);
  await ui.get('adviceButton').dispatch('click');assert.equal(ui.adviceRequests.length,1);
});

test('edits during simulation stop AI generation until the latest input is submitted',async()=>{
  const ui=await frontend(),release=ui.defer();await ui.edit('state_wind_m_s',10);
  const pending=ui.get('adviceButton').dispatch('click');assert.equal(ui.get('adviceButton').disabled,true);
  await ui.edit('state_wind_m_s',11);release();await pending;
  assert.equal(ui.adviceRequests.length,0);assert.match(ui.get('adviceContent').textContent,/参数已修改/);
  assert.equal(ui.get('adviceButton').disabled,false);
});

test('edits during AI requests discard old explanations and duplicate clicks do not send requests',async()=>{
  const ui=await frontend(),release=ui.deferAdvice();
  const pending=ui.get('adviceButton').dispatch('click');assert.equal(ui.get('adviceButton').disabled,true);
  await ui.get('adviceButton').dispatch('click');assert.equal(ui.adviceRequests.length,1);
  await ui.edit('state_wind_m_s',9.8);assert.equal(ui.get('adviceButton').disabled,true);
  release();await pending;assert.doesNotMatch(ui.get('adviceContent').textContent,/根据计算结果/);
  assert.match(ui.get('adviceContent').textContent,/参数已修改/);assert.equal(ui.get('adviceButton').disabled,false);
  await ui.get('adviceButton').dispatch('click');assert.equal(ui.adviceRequests.at(-1).state.wind_m_s,9.8);
});
