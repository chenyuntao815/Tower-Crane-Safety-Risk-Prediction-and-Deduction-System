const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const colors = ['#60a5fa','#34d399','#c4a4ff','#fb9678'];
const fmt = (v, digits=1) => v===null || v===undefined ? '—' : Number(v).toFixed(digits);
const pct = v => `${fmt(v*100)}%`;
let defaults, result, selected=0, index=0, timer=null, sceneView=null, runSequence=0, dirty=false;
let simulationInFlight=false, adviceInFlight=false;
const fields = {
  stateFields:['state',[
    ['slew_deg','回转角','°'],['slew_rate_deg_s','回转角速度','°/s'],['radius_m','幅度','m'],['load_t','吊重','t'],
    ['rope_length_m','绳长','m'],['wind_m_s','风速','m/s'],['wind_direction_deg','风向','°'],['swing_x_deg','X 向摆角','°'],
    ['swing_z_deg','Z 向摆角','°'],['swing_rate_x_deg_s','X 摆角速度','°/s'],['swing_rate_z_deg_s','Z 摆角速度','°/s']]],
  observationFields:['state',[
    ['mast_x_mm','X 塔顶位移','mm','optional'],['mast_z_mm','Z 塔顶位移','mm','optional'],
    ['mast_rate_x_mm_s','X 位移速度','mm/s'],['mast_rate_z_mm_s','Z 位移速度','mm/s']]],
  equipmentFields:['equipment',[
    ['tower_height_m','塔高','m'],['jib_length_m','臂长','m'],['attachment_heights_m','附着高度','m，逗号分隔','array'],
    ['mast_ei_kn_m2','塔身 EI','kN·m²'],['mast_frequency_hz','塔身主频','Hz'],['mast_damping','塔身阻尼比',''],
    ['swing_damping','摆动阻尼比',''],['tower_wind_area_m2','塔机迎风面积','m²'],['payload_wind_area_m2','吊物迎风面积','m²'],
    ['drag_coefficient','风阻系数',''],['attachment_lever_arm_m','附着等效力臂','m'],['rated_load_t','额定吊重','t'],
    ['rated_lift_moment_kn_m','额定起重力矩','kN·m'],['max_free_height_m','自由高度限值','m'],['payload_radius_m','吊物包络半径','m']]],
  uncertaintyFields:['uncertainty',[
    ['gust_std_m_s','阵风标准差 σ','m/s'],['gust_correlation_s','阵风相关时间','s'],['wind_bias_m_s','风速测量误差 ±','m/s'],
    ['wind_direction_error_deg','风向误差 ±','°'],['swing_error_deg','摆角误差 ±','°'],['swing_rate_error_deg_s','摆速误差 ±','°/s'],
    ['slew_error_deg','回转角误差 ±','°'],['slew_rate_error_deg_s','角速度误差 ±','°/s'],['relative_parameter_error','参数相对误差 ±','比例'],
    ['mast_prior_std_mm','位移先验标准差','mm'],['mast_measurement_std_mm','位移测量标准差','mm']]],
  limitFields:['limits',[
    ['swing_deg','摆角限值','°'],['swing_source','摆角依据','','text'],['mast_moment_kn_m','弯矩限值','kN·m'],['mast_source','弯矩依据','','text'],
    ['attachment_axial_kn','附着轴力限值','kN'],['attachment_source','轴力依据','','text'],['min_clearance_m','最小净距','m'],['clearance_source','距离依据','','text']]],
  obstacleFields:['obstacle',[
    ['enabled','启用障碍物','','bool'],['x_m','障碍物 X','m'],['z_m','障碍物 Z','m'],['radius_m','障碍物半径','m']]],
  controlFields:['controls',[
    ['early_deceleration_s','提前减速时长','s'],['reduced_speed_fraction','降速后速度比例','比例'],['smooth_braking_s','平缓制动时长','s'],
    ['immediate_braking_s','立即制动时长','s'],['max_acceleration_deg_s2','正常最大角减速度','°/s²'],['max_jerk_deg_s3','正常最大加加速度','°/s³'],
    ['emergency_acceleration_deg_s2','应急最大角减速度','°/s²'],['emergency_jerk_deg_s3','应急最大加加速度','°/s³'],['emergency_enabled','允许应急制动','','bool'],
    ['remaining_slew_deg','剩余回转角','°'],['restart_delay_s','重启等待时间','s'],['time_cost_per_s','时间代价','单位/s'],
    ['stop_cost','一次停机代价','单位'],['risk_cost','联合超限损失权重','单位'],['risk_budget','风险预算','0–1']]],
  runFields:['', [['horizon_s','预测时长','4–20 s'],['samples','共同样本数','64–1024'],['seed','随机种子','整数']]]
};
function fieldId(group,key){return `${group || 'run'}_${key}`}
function buildForm(config){
  for(const [id,[group,definitions]] of Object.entries(fields)){
    $(id).innerHTML=definitions.map(([key,label,unit,type])=>{
      const fid=fieldId(group,key),v=group?config[group][key]:config[key];
      const inputType=type==='bool'?'checkbox':type==='array'||type==='text'?'text':'number';
      return `<div class="field ${['text','array'].includes(type)?'wide':''}"><label for="${fid}">${label}<span>${unit}</span></label><input id="${fid}" type="${inputType}" ${type==='bool'?(v?'checked':''):`value="${esc(v===null?'':Array.isArray(v)?v.join(','):v)}"`} ${inputType==='number'?'step="any"':''} ${type==='optional'?'placeholder="模型先验"':''} ${type!=='optional'&&type!=='bool'?'required':''}></div>`;
    }).join('');
  }
}
function readForm(){
  const config=structuredClone(defaults);
  for(const [, [group,definitions]] of Object.entries(fields)){
    for(const [key,label,,type] of definitions){
      const element=$(fieldId(group,key));let value;
      if(type==='bool')value=element.checked;
      else if(type==='text') {value=element.value.trim();if(!value)throw Error(`${label}不能为空`)}
      else if(type==='array'){
        value=element.value.trim()===''?[]:element.value.split(/[,，]/).map(v=>{
          if(!v.trim()||!Number.isFinite(Number(v)))throw Error('附着高度请填写数字并用逗号分隔');return Number(v);
        });
      }else if(type==='optional' && !element.value.trim())value=null;
      else{value=Number(element.value);if(!element.value.trim()||!Number.isFinite(value))throw Error(`请填写有效的${label}`)}
      if(group)config[group][key]=value;else config[key]=value;
    }
  }
  return config;
}
async function api(path,payload){
  const response=await fetch(`/api/v1/operations/${path}`,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});
  const body=await response.json();
  if(!response.ok||body.code!==0){
    const detail=body.detail;
    throw Error(typeof detail==='string'?detail:Array.isArray(detail)?detail.map(e=>`${e.loc.slice(1).join('.')}：${e.msg}`).join('；'):body.message||'请求失败');
  }
  return body.data;
}
function showError(error){$('error').hidden=false;$('error').textContent=error.message||String(error)}
function useManualInput(){
  $('preset').value='manual';
  $('loadPreset').textContent='载入当前输入';
}
function markInputChanged(message){
  useManualInput();dirty=true;syncAdviceButton();
  $('adviceContent').textContent='参数已修改。点击“生成 DeepSeek 解释”将先更新推演，再解释当前输入的计算结果。';
  $('runStatus').textContent=message||(result?'参数已修改：当前图表仍为上一次推演，请载入当前输入或重新运行。':'参数已修改，请载入当前输入或运行推演。');
}
function matchesCurrentInput(config){
  try{return JSON.stringify(readForm())===JSON.stringify(config)}catch{return false}
}
function syncAdviceButton(){
  const button=$('adviceButton');
  button.disabled=!defaults||simulationInFlight||adviceInFlight;
  button.textContent=adviceInFlight?'正在生成解释…':'生成 DeepSeek 解释 ↗';
  button.title=!defaults?'正在读取参数':simulationInFlight?'正在推演，请稍候':adviceInFlight?'正在生成解释，请稍候':dirty||!result?'先更新当前输入的推演，再生成解释':'解释当前计算结果';
}
function pause(){if(timer)clearInterval(timer);timer=null;$('playButton').textContent='▶ 播放'}
async function run(){
  pause();$('error').hidden=true;
  let config;
  try{config=readForm()}catch(e){showError(e);return}
  const sequence=++runSequence;
  simulationInFlight=true;$('runButton').disabled=true;syncAdviceButton();
  $('runStatus').textContent=`正在配对推演 ${config.samples} 条样本 × 四种操作…`;
  const start=performance.now();
  try{
    const value=await api('simulate',config);if(sequence!==runSequence)return;
    result=value;dirty=!matchesCurrentInput(config);selected=Math.max(0,result.scenarios.findIndex(s=>s.id===result.recommendation.action_id));index=0;
    $('scenarioSelect').innerHTML=result.scenarios.map((s,i)=>`<option value="${i}">${s.label}</option>`).join('');
    $('scenarioSelect').value=selected;$('scenarioSelect').disabled=false;
    $('timeSlider').max=result.scenarios[0].series.length-1;$('timeSlider').value=0;
    for(const id of ['timeSlider','playButton','exportResult','exportSeries'])$(id).disabled=false;
    $('ensembleLabel').textContent=`${config.samples} 样本 · seed ${config.seed}`;
    $('runStatus').textContent=dirty?'推演期间参数已修改：图表对应提交时的输入，请重新运行。':`已完成 · ${fmt((performance.now()-start)/1000)} s · 共同样本 ${result.ensemble.id}`;
    $('adviceContent').textContent='计算结果已更新。可按需生成 AI 定性解释。';
    showComparison();showEvidence();refresh();
    return result;
  }catch(e){if(sequence===runSequence){showError(e);$('runStatus').textContent='推演未完成。请检查参数或服务连接。'}}
  finally{if(sequence===runSequence){simulationInFlight=false;$('runButton').disabled=false;syncAdviceButton()}}
}
function showComparison(){
  const rec=result.recommendation;
  $('decisionStatus').textContent={within_budget:'满足设定风险预算',budget_unmet:'未达到风险预算',no_feasible_action:'无可行候选'}[rec.status];
  $('recommendation').textContent=rec.text;$('recommendation').classList.toggle('alert',rec.status!=='within_budget');
  $('scenarioCards').innerHTML=result.scenarios.map((s,i)=>`<article class="scenario-card" data-index="${i}" tabindex="0" role="button" aria-label="查看${s.label}"><h3 style="color:${colors[i]}">${s.label}</h3><strong>${pct(s.risk.probability)}</strong><p>整个预测时段内任一超限</p><p>${s.comparison.risk_reduction_pp>=0?'降低':'增加'} ${fmt(Math.abs(s.comparison.risk_reduction_pp))} 个百分点</p><p>预计任务 ${fmt(s.costs.estimated_completion_s)} s</p><div class="card-note">${s.id===rec.action_id?(rec.status==='within_budget'?'计算器建议':'最低风险供复核'):s.constraints.feasible?'操作约束满足':'操作约束不满足'}</div></article>`).join('');
  $('comparisonTable').querySelector('tbody').innerHTML=result.scenarios.map(s=>`<tr><td>${s.label}</td><td>${pct(s.risk.probability)} / ${s.risk.ci95.map(pct).join('–')}</td><td>${fmt(s.comparison.risk_reduction_pp)} pp / ${s.comparison.paired_delta_ci95_pp.map(x=>fmt(x)).join('–')}</td><td>${fmt(s.peaks.swing_deg.p95)}°</td><td>${fmt(s.peaks.mast_moment_kn_m.p95,0)}</td><td>${fmt(s.peaks.attachment_axial_kn.p95,0)}</td><td>${fmt(s.costs.estimated_completion_s)} s</td><td>${fmt(s.costs.stop_cost,0)} / ${fmt(s.costs.total,0)}</td><td>${esc(s.constraints.feasible?'满足':s.constraints.reasons.join('；'))}${s.validity.small_angle_valid?'':' · 小角度模型不适用'}</td></tr>`).join('');
}
function showEvidence(){
  $('graphSource').textContent=result.graph_source+'；CAUSES 关系仅解释风险链，不参与概率计算。'+result.measure_knowledge_source+'。';
  $('knowledgeChains').innerHTML=result.knowledge_chains.map(x=>`<span class="chain">${esc(x.factor_name)} → ${esc(x.risk_name)}</span>`).join('');
  const l=result.config.limits;
  $('limitEvidence').innerHTML=[['摆角',l.swing_deg,'°',l.swing_source],['塔身弯矩',l.mast_moment_kn_m,'kN·m',l.mast_source],['附着轴力',l.attachment_axial_kn,'kN',l.attachment_source],['安全净距',l.min_clearance_m,'m',l.clearance_source]].map(([name,v,unit,source])=>`<div>${name}限值 ${v} ${unit} · ${esc(source)}</div>`).join('');
  $('modelWarnings').innerHTML=[...result.warnings,result.uncertainty_note,result.recommendation.rule].map(x=>`<p>${esc(x)}</p>`).join('');
  $('estimationInfo').innerHTML=`<p>状态：${esc(result.estimation.swing_state)}</p><p>塔顶位移后验标准差：X ${fmt(result.estimation.mast_posterior_std_mm[0])} / Z ${fmt(result.estimation.mast_posterior_std_mm[1])} mm</p><p>${esc(result.estimation.parameter_status)}</p><p>${esc(result.estimation.wind_process)}</p>`;
}
function refresh(){
  if(!result)return;
  const s=result.scenarios[selected],point=s.series[index],c=result.config;
  $('timeLabel').textContent=`${fmt(point.time_s)} / ${c.horizon_s} s`;
  $('metricCards').innerHTML=[['摆角',fmt(point.swing_deg.mean),'°'],['塔身弯矩',fmt(point.mast_moment_kn_m.mean,0),'kN·m'],['附着轴力',fmt(point.attachment_axial_kn.mean,0),'kN'],['累计超限概率',pct(point.cumulative_probability),`瞬时 ${pct(point.instantaneous_probability)}`]].map(([label,v,unit],i)=>`<div class="metric-card ${i===3?'accent':''}"><small>${label}</small><strong>${v}</strong><em>${unit}</em></div>`).join('');
  drawBand('chartSwing',s.series,'swing_deg',c.limits.swing_deg,colors[selected]);
  drawBand('chartMoment',s.series,'mast_moment_kn_m',c.limits.mast_moment_kn_m,colors[selected]);
  drawBand('chartAxial',s.series,'attachment_axial_kn',c.limits.attachment_axial_kn,colors[selected]);
  drawRisk();drawTrajectory();
  $('eventProbabilities').innerHTML=Object.entries(result.event_names).map(([key,label])=>`<div class="event-row"><span>${label}${key==='clearance'&&!c.obstacle.enabled?'（停用）':''}${key==='attachment'&&!c.equipment.attachment_heights_m.length?'（无附着）':''}</span><b>${pct(point.event_probabilities[key])}</b></div>`).join('');
  const explanations={swing:'回转加减速、已有摆动与吊物风载共同作用；急刹可能激发摆动。',moment:'吊重力矩、塔机风载、吊物水平反力与塔身主模态响应。',attachment:'由塔身弯矩、附着等效力臂和附着数量估计。',clearance:'吊物未来水平位置与障碍物包络的最小净距。'};
  $('eventReasons').innerHTML=Object.entries(s.risk.events).filter(([key])=>key!=='clearance'||c.obstacle.enabled).sort((a,b)=>b[1].probability-a[1].probability).map(([key,v])=>`<div class="event-row"><span>${result.event_names[key]} · ${explanations[key]}</span><b>${pct(v.probability)}</b></div>`).join('');
  $('costBreakdown').innerHTML=`<div class="cost-row"><strong>${s.label} · 代价组成</strong><span>风险 ${fmt(s.costs.risk_cost)} + 时间 ${fmt(s.costs.time_cost)} + 停机 ${fmt(s.costs.stop_cost)} = ${fmt(s.costs.total)}</span><span>峰值角减速度 ${fmt(s.constraints.peak_acceleration_deg_s2,2)} °/s² · 加加速度 ${fmt(s.constraints.peak_jerk_deg_s3,2)} °/s³</span><span>预测期内${s.costs.completion_within_horizon_s===null?'任务未完成':`到达目标 ${fmt(s.costs.completion_within_horizon_s)} s`} · 剩余 ${fmt(s.costs.remaining_slew_deg)}°</span></div>`;
  $('costBreakdown').insertAdjacentHTML('beforeend',`<p class="muted">适用条件：${esc(s.knowledge_rule.applies_when.join('；'))}。${esc(s.knowledge_rule.tradeoff)}</p>`);
  document.querySelectorAll('.scenario-card').forEach((card,i)=>card.classList.toggle('selected',i===selected));
  sceneView?.update(result,selected,index,$('showAllPaths').checked,$('amplifyMast').checked);
}
function line(points){return points.map(([x,y],i)=>`${i?'L':'M'}${fmt(x,2)},${fmt(y,2)}`).join(' ')}
function text(x,y,value,anchor='start'){return `<text x="${x}" y="${y}" fill="#859fae" font-size="8" text-anchor="${anchor}">${esc(value)}</text>`}
function drawBand(id,series,key,limit,color){
  const w=300,h=115,left=38,top=9,bottom=94,right=287;
  const max=Math.max(limit,...series.map(p=>p[key].p95))*1.14 || 1;
  const x=t=>left+(right-left)*t/result.config.horizon_s,y=v=>bottom-(bottom-top)*v/max;
  const upper=series.map(p=>[x(p.time_s),y(p[key].p95)]),lower=series.map(p=>[x(p.time_s),y(p[key].p05)]).reverse();
  $(id).innerHTML=`<path d="M${left},${top}V${bottom}H${right}" stroke="#38505e" fill="none"/>${[0,.5,1].map(f=>text(left-4,y(max*f)+3,fmt(max*f,max<20?1:0),'end')).join('')}<path d="${line([...upper,...lower])}Z" fill="${color}" opacity=".12"/><path d="${line(series.map(p=>[x(p.time_s),y(p[key].mean)]))}" stroke="${color}" stroke-width="1.8" fill="none"/><path d="M${left},${y(limit)}H${right}" stroke="#d6a371" stroke-dasharray="4,4" opacity=".8"/>${text(right,y(limit)-3,'限值','end')}<path d="M${x(series[index].time_s)},${top}V${bottom}" stroke="#dcebe9" opacity=".4"/>${text(left,108,'0 s')}${text(right,108,result.config.horizon_s+' s','end')}`;
}
function drawRisk(){
  const x=t=>37+250*t/result.config.horizon_s,y=p=>109-p*96;
  $('chartRisk').innerHTML=`<path d="M37,13V109H287" stroke="#38505e" fill="none"/>${[0,.5,1].map(v=>text(33,y(v)+3,Math.round(v*100),'end')).join('')}${result.scenarios.map((s,i)=>`<path d="${line(s.series.map(p=>[x(p.time_s),y(p.cumulative_probability)]))}" stroke="${colors[i]}" stroke-width="${i===selected?2.5:1.1}" opacity="${i===selected?1:.65}" fill="none"/>`).join('')}<path d="M37,${y(result.config.controls.risk_budget)}H287" stroke="#d6a371" stroke-dasharray="3,3"/>${text(285,y(result.config.controls.risk_budget)-3,'预算','end')}${text(37,125,'0 s')}${text(287,125,result.config.horizon_s+' s','end')}`;
}
function drawTrajectory(){
  const cases=$('showAllPaths').checked?result.scenarios:[result.scenarios[selected]],o=result.config.obstacle;
  const all=cases.flatMap(s=>s.trajectory_samples_xz_m.flat());
  if(o.enabled)all.push([o.x_m-o.radius_m-2,o.z_m-o.radius_m-2],[o.x_m+o.radius_m+2,o.z_m+o.radius_m+2]);
  const xmin=Math.min(...all.map(p=>p[0]))-2,xmax=Math.max(...all.map(p=>p[0]))+2,zmin=Math.min(...all.map(p=>p[1]))-2,zmax=Math.max(...all.map(p=>p[1]))+2;
  const scale=Math.min(540/(xmax-xmin),145/(zmax-zmin)),x=v=>55+(540-(xmax-xmin)*scale)/2+(v-xmin)*scale,z=v=>160-(v-zmin)*scale;
  const grid=[0,.25,.5,.75,1].map(f=>{const xv=xmin+(xmax-xmin)*f,zv=zmin+(zmax-zmin)*f;return `<path d="M${x(xv)},10V160M55,${z(zv)}H595" stroke="#263d4c"/>${text(x(xv),177,fmt(xv),'middle')}${text(49,z(zv)+3,fmt(zv),'end')}`}).join('');
  $('trajectoryChart').innerHTML=grid+text(604,177,'X')+text(38,11,'Z')+(o.enabled?`<circle cx="${x(o.x_m)}" cy="${z(o.z_m)}" r="${o.radius_m*scale}" fill="#7c5440" opacity=".45"/><circle cx="${x(o.x_m)}" cy="${z(o.z_m)}" r="${(o.radius_m+result.config.limits.min_clearance_m+result.config.equipment.payload_radius_m)*scale}" stroke="#bb8c63" stroke-dasharray="4,4" fill="none"/>`:'')+cases.map(s=>{
    const color=colors[result.scenarios.indexOf(s)];return s.trajectory_samples_xz_m.map(path=>`<path d="${line(path.map(p=>[x(p[0]),z(p[1])]))}" stroke="${color}" opacity=".12" fill="none"/>`).join('')+`<path d="${line(s.series.map(p=>[x(p.load_x_m.mean),z(p.load_z_m.mean)]))}" stroke="${color}" stroke-dasharray="5,3" fill="none"/><circle cx="${x(s.series[index].load_x_m.mean)}" cy="${z(s.series[index].load_z_m.mean)}" r="3" fill="${color}"/>`;
  }).join('');
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csvLine(values){return values.map(v=>`"${String(v??'').replaceAll('"','""')}"`).join(',')}
function exportSeries(){
  const headers=['action','time_s','slew_deg','slew_rate_deg_s','swing_deg','swing_p05','swing_p95','mast_moment_kn_m','moment_p05','moment_p95','attachment_axial_kn','axial_p05','axial_p95','load_x_m','load_z_m','mast_x_mm','mast_z_mm','instantaneous_probability','cumulative_probability','swing_exceedance','moment_exceedance','attachment_exceedance','clearance_exceedance'];
  const rows=result.scenarios.flatMap(s=>s.series.map(p=>[s.id,p.time_s,p.slew_deg,p.slew_rate_deg_s,p.swing_deg.mean,p.swing_deg.p05,p.swing_deg.p95,p.mast_moment_kn_m.mean,p.mast_moment_kn_m.p05,p.mast_moment_kn_m.p95,p.attachment_axial_kn.mean,p.attachment_axial_kn.p05,p.attachment_axial_kn.p95,p.load_x_m.mean,p.load_z_m.mean,p.mast_x_mm.mean,p.mast_z_mm.mean,p.instantaneous_probability,p.cumulative_probability,...Object.values(p.event_probabilities)]));
  download('crane-operation-series.csv','\ufeff'+[headers,...rows].map(csvLine).join('\r\n'),'text/csv;charset=utf-8');
}
function parseCsv(source){
  const lines=[];let row=[],cell='',quoted=false;
  source=source.replace(/^\ufeff/,'');
  for(let i=0;i<source.length;i++){
    const ch=source[i];if(ch==='"'){if(quoted&&source[i+1]==='"'){cell+='"';i++}else quoted=!quoted}
    else if(ch===','&&!quoted){row.push(cell);cell=''}
    else if((ch==='\n'||ch==='\r')&&!quoted){if(ch==='\r'&&source[i+1]==='\n')i++;row.push(cell);if(row.some(v=>v.trim()))lines.push(row);row=[];cell=''}else cell+=ch;
  }
  if(quoted)throw Error('CSV 引号未闭合');row.push(cell);if(row.some(v=>v.trim()))lines.push(row);
  const columns=(lines.shift()||[]).map(v=>v.trim());
  if(new Set(columns).size!==columns.length)throw Error('CSV 列名不能重复');
  return lines.map((values,i)=>{
    if(values.length!==columns.length)throw Error(`CSV 第${i+2}行列数不一致`);
    return Object.fromEntries(columns.map((key,j)=>{const raw=values[j].trim();const number=raw===''?null:Number(raw);if(number!==null&&!Number.isFinite(number))throw Error(`CSV 第${i+2}行 ${key} 不是有效数字`);return [key,number]}));
  });
}
async function validateFile(file){
  if(!file)return;if(file.size>5*1024*1024)throw Error('验证 CSV 请控制在5 MB内');
  $('validationResult').textContent='正在核验…';$('reliabilityChart').hidden=true;
  const report=await api('validate',{rows:parseCsv(await file.text())});
  $('validationResult').innerHTML=`<div class="validation-summary"><span>独立时段 ${report.runs}</span><span>Brier ${fmt(report.brier_score,4)}</span><span>ECE ${fmt(report.ece,4)}</span></div>`+Object.entries(report.prediction_error).map(([key,v])=>`<p>${esc(key)}：n=${v.count}，MAE ${fmt(v.mae,3)}，RMSE ${fmt(v.rmse,3)}，90%区间覆盖 ${v.p05_p95_coverage===null?'未提供区间':pct(v.p05_p95_coverage)}（n=${v.interval_count}）</p>`).join('')+`<p>${esc(report.status)}。${esc(report.note)}</p>`;
  if(report.reliability_bins.length){
    const x=v=>55+535*v,y=v=>184-164*v;
    $('reliabilityChart').hidden=false;
    $('reliabilityChart').innerHTML=`<path d="M55,20V184H590M55,184L590,20" stroke="#465c6a" fill="none" stroke-dasharray="4,4"/>${[0,.25,.5,.75,1].map(v=>text(x(v),201,pct(v),'middle')+text(46,y(v)+3,pct(v),'end')).join('')}${text(322,217,'预测概率','middle')}${text(55,10,'实测超限频率 / Wilson 95%区间')}`+report.reliability_bins.map(b=>`<path d="M${x(b.mean_predicted)},${y(b.ci95[0])}V${y(b.ci95[1])}" stroke="#a3dfc8"/><circle cx="${x(b.mean_predicted)}" cy="${y(b.observed_frequency)}" r="4" fill="#a3dfc8"/>${text(x(b.mean_predicted)+7,y(b.observed_frequency)-5,'n='+b.count)}`).join('');
  }
}
function choose(i){selected=i;$('scenarioSelect').value=i;refresh()}
$('operationForm').addEventListener('submit',e=>{e.preventDefault();run()});
$('operationForm').addEventListener('input',()=>markInputChanged());
$('scenarioSelect').addEventListener('change',e=>choose(Number(e.target.value)));
$('scenarioCards').addEventListener('click',e=>{const card=e.target.closest('[data-index]');if(card)choose(Number(card.dataset.index))});
$('scenarioCards').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){const card=e.target.closest('[data-index]');if(card){e.preventDefault();choose(Number(card.dataset.index))}}});
$('timeSlider').addEventListener('input',e=>{index=Number(e.target.value);refresh()});
for(const id of ['showAllPaths','amplifyMast'])$(id).addEventListener('change',refresh);
$('playButton').addEventListener('click',()=>{if(!result)return;if(timer){pause();return}if(index>=result.scenarios[0].series.length-1)index=0;$('playButton').textContent='Ⅱ 暂停';timer=setInterval(()=>{index++;$('timeSlider').value=index;refresh();if(index>=result.scenarios[0].series.length-1)pause()},200)});
$('preset').addEventListener('change',()=>{
  $('loadPreset').textContent=$('preset').value==='manual'?'载入当前输入':'填入所选示例';
});
$('loadPreset').addEventListener('click',async()=>{
  if(!defaults)return;
  const preset=$('preset').value;
  if(preset==='manual'){await run();return}
  const c=structuredClone(defaults),label=$('preset').selectedOptions[0].textContent;
  if(preset==='obstacle'){c.obstacle={enabled:true,x_m:10,z_m:33,radius_m:2};c.limits.attachment_axial_kn=280}
  if(preset==='swing'){c.state.swing_x_deg=3;c.state.swing_z_deg=-2;c.state.swing_rate_z_deg_s=.8;c.limits.attachment_axial_kn=280}
  if(preset==='gust'){c.state.wind_m_s=18;c.uncertainty.gust_std_m_s=4;c.limits.mast_moment_kn_m=2300;c.limits.attachment_axial_kn=220}
  if(preset==='overload'){c.state.load_t=15}
  // Replacing the form is reserved for an explicitly selected example.
  // Return to manual mode so subsequent loads use the edited values.
  pause();runSequence++;simulationInFlight=false;$('runButton').disabled=false;$('error').hidden=true;
  buildForm(c);markInputChanged(`已填入${label}参数，可继续修改。点击“载入当前输入”或“运行四种操作推演”开始计算。`);
});
$('exportConfig').addEventListener('click',()=>{try{download('crane-input.json',JSON.stringify(readForm(),null,2),'application/json')}catch(e){showError(e)}});
$('importConfig').addEventListener('change',async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>100000)throw Error('参数文件过大');const c=JSON.parse(await file.text());if(!c.state||!c.controls||!c.equipment||!c.uncertainty||!c.limits||!c.obstacle)throw Error('请导入本平台导出的完整参数文件');buildForm(c);useManualInput();await run()}catch(e){showError(e)}finally{e.target.value=''}});
$('exportResult').addEventListener('click',()=>result&&download('crane-operation-result.json',JSON.stringify(result,null,2),'application/json'));
$('exportSeries').addEventListener('click',exportSeries);
$('validationTemplate').addEventListener('click',()=>download('crane-validation-template.csv','\ufeff'+csvLine(['risk_probability','observed_exceedance',...['swing_deg','mast_moment_kn_m','attachment_axial_kn'].flatMap(key=>['pred_'+key,'observed_'+key,'p05_'+key,'p95_'+key])])+'\r\n','text/csv;charset=utf-8'));
$('validationFile').addEventListener('change',async e=>{try{await validateFile(e.target.files[0])}catch(error){$('validationResult').textContent='验证失败：'+error.message}finally{e.target.value=''}});
$('adviceButton').addEventListener('click',async()=>{
  if(!defaults||simulationInFlight||adviceInFlight)return;
  adviceInFlight=true;syncAdviceButton();
  let current;
  try{
    if(!result||dirty||!matchesCurrentInput(result.config)){
      $('adviceContent').textContent='正在更新当前输入的推演…';
      current=await run();
      if(!current){$('adviceContent').textContent='AI 解释未生成：请先修正输入或解决推演错误，再点击重试。';return}
    }else current=result;
    if(result!==current||!matchesCurrentInput(current.config)){
      $('adviceContent').textContent='参数已修改，请再次点击生成解释，以使用最新输入。';return;
    }
    $('adviceContent').textContent='正在根据服务端重算结果生成文字解释…';
    const response=await api('advice',current.config);
    // Ignore stale explanations when another simulation or edit occurred during the request.
    if(result!==current||!matchesCurrentInput(current.config))return;
    if(response.ensemble_id!==current.ensemble.id)throw Error('解释与当前推演样本不一致，请重新生成');
    const d=response.explanation;
    $('adviceContent').textContent=[d.situation_analysis,...d.main_warnings.map(x=>'风险：'+x),...d.prevention_measures.map(x=>'措施：'+x),'应急：'+d.emergency_response,'监测：'+d.monitoring_focus].join('\n');
  }catch(e){
    if(!current||(result===current&&matchesCurrentInput(current.config)))$('adviceContent').textContent='AI 解释未生成：'+e.message+'。计算建议与比较结果仍可使用。';
  }finally{adviceInFlight=false;syncAdviceButton()}
});
// Load the renderer independently: unsupported WebGL must not disable the calculator.
import('./operation_scene.mjs').then(module=>{sceneView=module.createScene($('scene'),message=>{$('inspector').textContent=message});$('sceneStatus').hidden=true;if(result)refresh()}).catch(e=>{$('sceneStatus').textContent='三维视图不可用：'+e.message+'。可继续使用下方俯视轨迹及全部计算功能。'});
try{defaults=await api('defaults');buildForm(defaults);await run()}catch(e){showError(e);$('runStatus').textContent='请启动本项目 API 服务后刷新页面。'}
