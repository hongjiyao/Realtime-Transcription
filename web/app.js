// P-18: 应用状态模块化 — 按职责分组为子命名空间
const AppState = {
  // 连接管理
  connection: {
    ws: null,
    reconnectDelay: 1000,
    maxReconnectDelay: 30000,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    reconnectTimer: null,
  },
  // 转录数据
  transcription: {
    isTranscribing: false,
    waitingForStop: false,
    _isStarting: false,
    hotwords: '',
    currentPartial: '',
    finalSentences: [],
    _renderedSentenceCount: 0,
    _scrollPending: false,
    _partialScrollPending: false,
    _renderHtmlCache: new Map(),
    latencyHistory: {1: [], 2: []},
  },
  // UI 定时器与缓存
  ui: {
    _configSyncTimer: null,
    _successTimer: null,
    _errorTimer: null,
    _modelData: {},
  },
};

function wsToHttp(wsUrl) {
    return wsUrl.replace(/\/+$/, '').replace(/^ws/, 'http').replace(/\/ws$/, '');
}

function showEl(id, cls) {
    const el = document.getElementById(id);
    if (el) { el.classList.remove('hidden'); el.classList.add(cls); }
}
function hideEl(id) {
    const el = document.getElementById(id);
    if (el) { el.classList.remove('show-inline', 'show-inline-block', 'show-inline-flex', 'show-grid'); el.classList.add('hidden'); }
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-page').forEach(function(p) {
    p.classList.toggle('active', p.id === 'page-' + tabName);
  });
}

const speakerColors = ['#64ffda','#f472b6','#fbbf24','#34d399','#a78bfa','#fb923c','#67e8f9','#f87171','#38bdf8','#c084fc'];

const CONFIG_ITEMS = [
    { id: 'maxSentenceChars', key: 'max_sentence_chars', def: 100, fmt: v => v },
    { id: 'vadSilenceGap', key: 'vad_silence_gap_sec', def: 0.5, fmt: v => v },
    { id: 'vadCheckInterval', key: 'vad_check_interval_sec', def: 0.15, fmt: v => v },
    { id: 'presetSpkNum', key: 'preset_spk_num', def: 2, fmt: v => v == 1 ? '自动' : v },
    { id: 'spkSimThreshold', key: 'spk_sim_threshold', def: 0.65, fmt: v => parseFloat(v).toFixed(2) },
    { id: 'spkEmaAlpha', key: 'spk_ema_alpha', def: 0.3, fmt: v => parseFloat(v).toFixed(2) },
    { id: 'chunkSizeFrames', key: 'chunk_size_frames', def: 5, fmt: v => v + '帧 (' + (v * 60) + 'ms)' },
];

// 参数自动同步防抖

function onModelChange() {
  const modelSelect = document.getElementById('offlineModel');
  const langSelect = document.getElementById('language');
  if (!modelSelect || !langSelect) return;
  const modelKey = modelSelect.value;
  const modelData = AppState.ui._modelData[modelKey] || {};
  // 更新语言选项
  const langOptions = modelData.lang_options || [{value: "auto", label: "Auto"}];
  langSelect.innerHTML = '';
  for (let i = 0; i < langOptions.length; i++) {
    const opt = document.createElement('option');
    opt.value = langOptions[i].value;
    opt.textContent = langOptions[i].label;
    langSelect.appendChild(opt);
  }
  // 更新说话人分离区域
  const spkSection = document.getElementById('speakerParams');
  const spkCheckbox = document.getElementById('speakerDiarization');
  if (Object.keys(AppState.ui._modelData).length > 0 && !modelData.supports_spk) {
    // 当前模型不支持说话人分离（仅在模型数据已加载时判断）
    if (spkCheckbox.checked) {
      spkCheckbox.checked = false;
      showError('当前模型不支持说话人分离，已自动关闭。如需说话人分离，请选择 SeACoParaformer 模型。', false);
    }
    spkSection.classList.add('hidden');
    spkSection.classList.add('disabled');
  } else if (modelData.supports_spk) {
    spkSection.classList.remove('disabled');
  }
  // 更新热词输入
  const hwText = document.getElementById('hotwordText');
  if (Object.keys(AppState.ui._modelData).length > 0 && !modelData.supports_hotwords) {
    hwText.disabled = true;
    hwText.style.opacity = '0.4';
  } else {
    hwText.disabled = false;
    hwText.style.opacity = '1';
  }
  // 更新模型延迟信息
  const latencyInfo = modelData.latency_info;
  const infoEl = document.getElementById('modelInfo');
  if (latencyInfo && infoEl) {
    infoEl.innerHTML = '<span class="badge ' + latencyInfo.badge + '"></span>';
    infoEl.querySelector('.badge').textContent = latencyInfo.desc;
  }
}

function onSpeakerDiarizationChange() {
  const spkCheckbox = document.getElementById('speakerDiarization');
  const spkParams = document.getElementById('speakerParams');
  if (!spkCheckbox || !spkParams) return;
  const modelSelect = document.getElementById('offlineModel');
  const currentModel = modelSelect.value;

  if (spkCheckbox.checked) {
    const caps = AppState.ui._modelData[currentModel] || {};
    if (!caps.supports_spk) {
      modelSelect.value = 'seaco_paraformer';
      onModelChange();
      showError('已自动切换到 SeACoParaformer 模型（唯一支持说话人分离的模型）', false);
    }
    spkParams.classList.remove('hidden');
  } else {
    spkParams.classList.add('hidden');
  }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
  if (!window.isSecureContext) {
    document.getElementById('secureWarning').classList.add('show');
  }
  initConfigPanel();
  onModelChange();
  checkDeviceStatus();
  loadAudioDevices();
  loadModelAvailability();
  initHotwordTextarea();

  // 按钮事件绑定
  document.getElementById('startBtn').addEventListener('click', startTranscription);
  document.getElementById('stopBtn').addEventListener('click', stopTranscription);
  document.getElementById('retryBtn').addEventListener('click', retryConnect);
  document.getElementById('errorBannerClose').addEventListener('click', hideError);
  document.getElementById('copyBtn').addEventListener('click', copyTranscription);
  document.getElementById('exportBtn').addEventListener('click', exportTranscription);
  document.getElementById('spkSaveBtn').addEventListener('click', saveSpeakerBank);
  document.getElementById('spkLoadBtn').addEventListener('click', loadSpeakerBankList);
  document.getElementById('spkLoadConfirmBtn').addEventListener('click', confirmLoadSpeakerBank);

  // Tab 切换
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      switchTab(this.dataset.tab);
    });
  });

  // 设置页联动
  document.getElementById('offlineModel').addEventListener('change', onModelChange);
  document.getElementById('speakerDiarization').addEventListener('change', onSpeakerDiarizationChange);
  document.getElementById('deviceSelect').addEventListener('change', onDeviceChange);
  document.getElementById('audioDeviceSelect').addEventListener('change', onAudioDeviceChange);
});

// 检查模型本地可用性
async function loadModelAvailability() {
  try {
    const response = await fetch('/models');
    const data = await response.json();
    const models = data.models || {};
    // 缓存模型能力数据
    AppState.ui._modelData = models;
    const select = document.getElementById('offlineModel');
    for (let i = 0; i < select.options.length; i++) {
      const opt = select.options[i];
      const key = opt.value;
      if (models[key] && !models[key].available) {
        opt.disabled = true;
        opt.textContent = opt.textContent + ' (未下载)';
      }
    }
    // 如果当前选中的模型不可用，自动切换到第一个可用的
    if (select.options[select.selectedIndex] && select.options[select.selectedIndex].disabled) {
      for (let i = 0; i < select.options.length; i++) {
        if (!select.options[i].disabled) {
          select.selectedIndex = i;
          onModelChange();
          break;
        }
      }
    }
    // 数据加载后刷新模型相关 UI
    onModelChange();
  } catch (e) {
    console.warn('Failed to check model availability:', e);
  }
}

// 检测设备状态
async function checkDeviceStatus() {
  try {
    const response = await fetch('/device/status');
    const data = await response.json();
    const dev = data.device || {};
    const infoEl = document.getElementById('deviceInfo');
    if (dev.cuda_available) {
      infoEl.innerHTML = '<span class="badge badge-fast">GPU 可用</span>';
      const devSpan = document.createElement('span');
      devSpan.textContent = (dev.gpu_name || 'Unknown GPU') + ' · ' + (dev.gpu_memory || '');
      infoEl.appendChild(devSpan);
      document.querySelector('#deviceSelect option[value="cuda"]').disabled = false;
    } else {
      infoEl.innerHTML = '<span class="badge badge-slow">GPU 不可用</span>';
      const noGpuSpan = document.createElement('span');
      noGpuSpan.textContent = 'PyTorch ' + (dev.torch_version || '') + ' · CUDA ' + (dev.cuda_version || 'N/A');
      infoEl.appendChild(noGpuSpan);
      document.querySelector('#deviceSelect option[value="cuda"]').disabled = true;
      document.querySelector('#deviceSelect option[value="cuda"]').textContent = 'GPU (CUDA) — 不可用';
    }
  } catch (e) {
    document.getElementById('deviceInfo').innerHTML = '<span class="badge badge-slow">检测失败</span>';
  }
}

function onDeviceChange() {
  const device = document.getElementById('deviceSelect').value;
  if (device === 'cuda') {
    const cudaOption = document.querySelector('#deviceSelect option[value="cuda"]');
    if (cudaOption && cudaOption.disabled) {
      showError('GPU (CUDA) 不可用，请先安装 CUDA 版本的 PyTorch', false);
      document.getElementById('deviceSelect').value = 'auto';
    }
  }
}

// 加载音频输出设备列表
async function loadAudioDevices() {
  const select = document.getElementById('audioDeviceSelect');
  const infoEl = document.getElementById('audioDeviceInfo');
  try {
    const response = await fetch('/audio-devices');
    const data = await response.json();
    if (!data.available) {
      infoEl.innerHTML = '<span class="badge badge-slow">WASAPI 不可用</span>';
      select.disabled = true;
      return;
    }
    const devices = data.devices || [];
    select.innerHTML = '<option value="auto">自动检测</option>';
    for (let i = 0; i < devices.length; i++) {
      const d = devices[i];
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name + ' (' + d.sample_rate + 'Hz, ' + d.channels + 'ch)';
      select.appendChild(opt);
    }
    if (devices.length > 0) {
      infoEl.innerHTML = '<span class="badge badge-fast"></span><span>选择要捕获的音频输出</span>';
      infoEl.querySelector('.badge').textContent = devices.length + ' 个设备';
    } else {
      infoEl.innerHTML = '<span class="badge badge-slow">未检测到设备</span><span>请确保有音频输出设备</span>';
    }
  } catch (e) {
    infoEl.innerHTML = '<span class="badge badge-slow">加载失败</span>';
  }
}

function onAudioDeviceChange() {
  const select = document.getElementById('audioDeviceSelect');
  const deviceIndex = select.value;
  if (AppState.connection.ws && AppState.connection.ws.readyState === WebSocket.OPEN && AppState.transcription.isTranscribing) {
    AppState.connection.ws.send('DEVICE_SELECT:' + deviceIndex);
  }
}

// 延迟显示相关
const LATENCY_HISTORY_SIZE = 20;

function updateLatency(pass, latency) {
  showEl('latencyPanel', 'show-grid');

  const valueEl = document.getElementById('pass' + pass + 'Latency');
  const rtfEl = document.getElementById('pass' + pass + 'Rtf');
  if (!valueEl || !latency) return;

  const inferenceMs = latency.inference_ms || 0;
  const rtf = latency.rtf || 0;

  AppState.transcription.latencyHistory[pass].push(inferenceMs);
  if (AppState.transcription.latencyHistory[pass].length > LATENCY_HISTORY_SIZE) {
    AppState.transcription.latencyHistory[pass].shift();
  }

  let avgMs = 0;
  const arr = AppState.transcription.latencyHistory[pass];
  for (let i = 0; i < arr.length; i++) avgMs += arr[i];
  avgMs = avgMs / arr.length;

  let speedClass = 'fast';
  if (pass === 1) {
    speedClass = avgMs < 50 ? 'fast' : avgMs < 150 ? 'medium' : 'slow';
  } else {
    speedClass = avgMs < 200 ? 'fast' : avgMs < 800 ? 'medium' : 'slow';
  }

  valueEl.textContent = avgMs.toFixed(0);
  valueEl.className = 'latency-value ' + speedClass;

  let rtfText = 'RTF: ' + rtf.toFixed(3);
  if (rtf < 1) {
    rtfText += ' (' + (1/rtf).toFixed(0) + 'x实时)';
  }
  rtfEl.textContent = rtfText;
}

function formatTime(ms) {
  let s = Math.floor(ms / 1000);
  let m = Math.floor(s / 60);
  s = s % 60;
  return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
}

// 配置面板相关函数
function initConfigPanel() {
  // 为每个滑块绑定事件
  CONFIG_ITEMS.forEach(function(item) {
    const el = document.getElementById(item.id);
    if (!el) return;
    // input 事件：实时更新显示值 + 保存 localStorage
    el.addEventListener('input', function(e) {
      const valEl = document.getElementById(item.id + 'Value');
      if (valEl) {
        const v = item.id === 'presetSpkNum' ? parseInt(e.target.value) :
                  item.id === 'maxSentenceChars' || item.id === 'chunkSizeFrames' ? parseInt(e.target.value) :
                  parseFloat(e.target.value);
        valEl.textContent = item.fmt(v);
      }
      saveConfigToLocal();
    });
    // change 事件：自动同步到后端（防抖 500ms）
    el.addEventListener('change', function() {
      debouncedSyncConfig();
    });
  });

  // 为非滑块控件绑定 change 事件，保存到 localStorage 并同步到后端
  ['serverUrl', 'audioDeviceSelect', 'deviceSelect', 'language', 'offlineModel'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', function() { saveConfigToLocal(); debouncedSyncConfig(); });
  });
  const spkDiarEl = document.getElementById('speakerDiarization');
  if (spkDiarEl) spkDiarEl.addEventListener('change', function() { saveConfigToLocal(); debouncedSyncConfig(); });

  loadConfig();
}

// 保存所有面板配置到 localStorage
function buildConfigObject() {
    return {
        max_sentence_chars: parseInt(document.getElementById('maxSentenceChars').value),
        vad_silence_gap_sec: parseFloat(document.getElementById('vadSilenceGap').value),
        vad_check_interval_sec: parseFloat(document.getElementById('vadCheckInterval').value),
        preset_spk_num: parseInt(document.getElementById('presetSpkNum').value),
        spk_sim_threshold: parseFloat(document.getElementById('spkSimThreshold').value),
        spk_ema_alpha: parseFloat(document.getElementById('spkEmaAlpha').value),
        chunk_size_frames: parseInt(document.getElementById('chunkSizeFrames').value),
        server_url: document.getElementById('serverUrl').value,
        audio_device: document.getElementById('audioDeviceSelect').value,
        device: document.getElementById('deviceSelect').value,
        language: document.getElementById('language').value,
        offline_model: document.getElementById('offlineModel').value,
        speaker_diarization: document.getElementById('speakerDiarization').checked,
        hotword_text: document.getElementById('hotwordText').value
    };
}

function saveConfigToLocal() {
  localStorage.setItem('asr_config', JSON.stringify(buildConfigObject()));
}

// 防抖同步配置到后端
function debouncedSyncConfig() {
  if (AppState.ui._configSyncTimer) clearTimeout(AppState.ui._configSyncTimer);
  AppState.ui._configSyncTimer = setTimeout(function() {
    syncConfigToServer();
  }, 500);
}

// 同步配置到后端
async function syncConfigToServer() {
  try {
    await fetch('/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildConfigObject())
    });
  } catch (e) {
    console.warn('Failed to sync config:', e);
  }
}

async function loadConfig() {
  // 优先从 localStorage 恢复，没有则从服务器加载
  let config = {};
  const saved = localStorage.getItem('asr_config');
  if (saved) {
    try { config = JSON.parse(saved); } catch(e) { config = {}; }
  }

  if (Object.keys(config).length === 0) {
    try {
      const response = await fetch('/config');
      config = await response.json();
    } catch (e) {
      console.error('Failed to load config:', e);
    }
  }

  // 应用到 UI（滑块参数）
  for (const item of CONFIG_ITEMS) {
    const val = config[item.key] !== undefined ? config[item.key] : item.def;
    const el = document.getElementById(item.id);
    const valEl = document.getElementById(item.id + 'Value');
    if (el) el.value = val;
    if (valEl) valEl.textContent = item.fmt(val);
  }

  // 恢复面板控件（连接设置 + 模型设置）
  if (config.server_url !== undefined) document.getElementById('serverUrl').value = config.server_url;
  if (config.audio_device !== undefined) document.getElementById('audioDeviceSelect').value = config.audio_device;
  if (config.device !== undefined) document.getElementById('deviceSelect').value = config.device;
  if (config.language !== undefined) document.getElementById('language').value = config.language;
  if (config.offline_model !== undefined) document.getElementById('offlineModel').value = config.offline_model;
  if (config.speaker_diarization !== undefined) {
    document.getElementById('speakerDiarization').checked = config.speaker_diarization;
    // 联动：勾选时展开说话人参数
    const spkParams = document.getElementById('speakerParams');
    if (config.speaker_diarization) {
      spkParams.classList.remove('hidden');
    }
  }
  if (config.hotword_text !== undefined) {
    document.getElementById('hotwordText').value = config.hotword_text;
    updateHotwordFromTextarea();
  }

  // 如果从 localStorage 恢复了配置，同步到服务器
  if (saved && Object.keys(config).length > 0) {
    try {
      await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
    } catch (e) { console.warn('[Config] 同步配置到服务器失败:', e); }
  }
}

// 热词文本框初始化
function initHotwordTextarea() {
  const textarea = document.getElementById('hotwordText');
  if (!textarea) return;
  textarea.addEventListener('input', function() {
    updateHotwordFromTextarea();
    saveConfigToLocal();
  });
}

// 从 textarea 解析热词
function updateHotwordFromTextarea() {
  const textarea = document.getElementById('hotwordText');
  const infoEl = document.getElementById('hotwordInfo');
  const text = textarea.value.trim();
  if (!text) {
    AppState.transcription.hotwords = '';
    infoEl.textContent = '未输入热词';
    infoEl.style.color = '';
    return;
  }
  // 逗号、换行、空格分隔
  const words = text.split(/[,，\n\r]+/).map(function(w) { return w.trim(); }).filter(function(w) { return w.length > 0; });
  AppState.transcription.hotwords = words.join(',');
  infoEl.textContent = '已输入 ' + words.length + ' 个热词';
  infoEl.style.color = '#64ffda';
}

function showSuccess(message) {
  const banner = document.getElementById('successBanner');
  document.getElementById('successMessage').textContent = message;
  banner.classList.add('show');
  if (AppState.ui._successTimer) clearTimeout(AppState.ui._successTimer);
  AppState.ui._successTimer = setTimeout(function() { banner.classList.remove('show'); }, 3000);
}

// P26: 合法 CSS 颜色白名单（仅允许 hex 格式和 transparent）
const SAFE_COLOR_RE = /^(#[0-9a-fA-F]{3,8}|transparent)$/;

function getSpeakerColor(spk) {
  let idx = 0;
  if (typeof spk === 'number') {
    idx = spk;
  } else if (typeof spk === 'string') {
    const match = spk.match(/\d+/);
    if (match) idx = parseInt(match[0], 10);
  }
  return speakerColors[idx % speakerColors.length];
}

function showError(msg, showRetry) {
  const banner = document.getElementById('errorBanner');
  const msgEl = document.getElementById('errorMessage');
  const retryBtn = document.getElementById('retryBtn');
  msgEl.textContent = msg;
  banner.classList.add('show');
  retryBtn.style.display = showRetry ? 'inline-flex' : 'none';
  // 非致命错误5秒后自动消失
  if (!showRetry) {
    if (AppState.ui._errorTimer) clearTimeout(AppState.ui._errorTimer);
    AppState.ui._errorTimer = setTimeout(function() { banner.classList.remove('show'); }, 5000);
  }
}

function hideError() {
  document.getElementById('errorBanner').classList.remove('show');
  if (AppState.ui._errorTimer) { clearTimeout(AppState.ui._errorTimer); AppState.ui._errorTimer = null; }
}

function setConnected(state) {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.classList.remove('connected', 'connecting');
  if (state === true) {
    dot.classList.add('connected');
    text.textContent = '已连接';
  } else if (state === 'connecting') {
    dot.classList.add('connecting');
    text.textContent = '连接中...';
  } else if (state === 'reconnecting') {
    dot.classList.add('connecting');
    text.textContent = '重连中 (第' + AppState.connection.reconnectAttempts + '次)...';
  } else {
    text.textContent = '未连接';
  }
}

function setTranscribing(val) {
  AppState.transcription.isTranscribing = val;
  if (val) {
    hideEl('startBtn');
    showEl('stopBtn', 'show-inline-flex');
  } else {
    showEl('startBtn', 'show-inline-flex');
    hideEl('stopBtn');
  }
}

function renderTranscription() {
  const area = document.getElementById('transcriptionArea');
  const empty = document.getElementById('emptyState');
  if (empty) empty.style.display = 'none';

  const startIdx = AppState.transcription._renderedSentenceCount;
  if (AppState.transcription.finalSentences.length > startIdx) {
    const fragment = document.createDocumentFragment();
    for (let i = startIdx; i < AppState.transcription.finalSentences.length; i++) {
      const s = AppState.transcription.finalSentences[i];
      const ms = s.start_ms || 0;
      const spk = s.spk !== undefined && s.spk !== null ? s.spk : -1;
      const color = spk >= 0 ? getSpeakerColor(spk) : 'transparent';
      const spkLabel = spk >= 0 ? 'SPK' + spk : '';
      // P26: 校验颜色值，不合法则回退到 transparent
      const safeColor = SAFE_COLOR_RE.test(color) ? color : 'transparent';

      // 优化：缓存 HTML 字符串，避免重复拼接
      const cacheKey = `${spk}-${ms}`;
      let html = AppState.transcription._renderHtmlCache.get(cacheKey);
      if (!html) {
        html = '<span class="timestamp">' + formatTime(ms) + '</span>';
        if (spkLabel) {
          html += '<span class="speaker" style="color:' + safeColor + ';background:' + safeColor + '18">' + escapeHtml(spkLabel) + '</span>';
        }
        if (AppState.transcription._renderHtmlCache.size < 50) {
          AppState.transcription._renderHtmlCache.set(cacheKey, html);
        }
      }
      html += '<span class="text-content">' + escapeHtml(s.text || '') + '</span>';
      
      const div = document.createElement('div');
      div.className = 'sentence final-sentence';
      div.innerHTML = html;
      fragment.appendChild(div);
    }
    area.appendChild(fragment);
    AppState.transcription._renderedSentenceCount = AppState.transcription.finalSentences.length;
  }

  // 节流滚动
  if (!AppState.transcription._scrollPending) {
    AppState.transcription._scrollPending = true;
    requestAnimationFrame(function() {
      area.scrollTop = area.scrollHeight;
      AppState.transcription._scrollPending = false;
    });
  }
}

function updatePartial() {
  let el = document.getElementById('pass1Text');
  if (!el) return;
  el.textContent = AppState.transcription.currentPartial || '';
  // 用 requestAnimationFrame 节流滚动，避免每次更新触发 reflow
  if (!AppState.transcription._partialScrollPending && AppState.transcription.currentPartial) {
    AppState.transcription._partialScrollPending = true;
    requestAnimationFrame(function() {
      let area = document.getElementById('pass1Area');
      if (area) area.scrollTop = area.scrollHeight;
      AppState.transcription._partialScrollPending = false;
    });
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function connectWebSocket(url) {
  return new Promise(function(resolve, reject) {
    setConnected('connecting');
    try {
      AppState.connection.ws = new WebSocket(url);
    } catch (e) {
      reject(e);
      return;
    }
    AppState.connection.ws.onopen = function() {
      setConnected(true);
      hideError();
      AppState.connection.reconnectDelay = 1000;
      resolve();
    };
    AppState.connection.ws.onerror = function(e) {
      reject(e);
    };
    AppState.connection.ws.onclose = function() {
      if (AppState.transcription.isTranscribing) {
        setConnected('reconnecting');
        showError('WebSocket 连接已断开，正在尝试重连...', true);
        if (AppState.connection.reconnectTimer) clearTimeout(AppState.connection.reconnectTimer);
        AppState.connection.reconnectTimer = setTimeout(function() {
          console.log('[INFO] Reconnecting WebSocket, delay=' + AppState.connection.reconnectDelay + 'ms');
          const baseUrl = document.getElementById('serverUrl').value.trim();
          connectWebSocket(baseUrl).then(function() {
            AppState.connection.reconnectAttempts = 0;
            if (AppState.transcription.isTranscribing) {
              startTranscription();
            }
          }).catch(function(e) {
            console.error('[ERROR] Reconnect failed:', e);
            AppState.connection.reconnectAttempts++;
            if (AppState.connection.reconnectAttempts >= AppState.connection.maxReconnectAttempts) {
              showError('重连失败次数过多，请检查服务器是否在线后刷新页面', false);
            }
          });
          AppState.connection.reconnectDelay = Math.min(AppState.connection.reconnectDelay * 2, AppState.connection.maxReconnectDelay);
        }, AppState.connection.reconnectDelay);
      }
    };
    AppState.connection.ws.onmessage = function(event) {
      handleMessage(event.data);
    };
  });
}

// ===== WebSocket 事件处理函数 =====

function handleError(msg) {
  showError('服务错误：' + (msg.message || '未知错误'), false);
}

function handleModelSwitched(msg) {
  showSuccess('已切换到 ' + (msg.name || msg.model));
  onModelChange();
  // 处理服务端兼容性警告（如自动关闭说话人分离）
  if (msg.warnings && msg.warnings.length > 0) {
    showError(msg.warnings.join('；'), false);
  }
  // 如果服务端自动关闭了说话人分离，同步前端 UI
  if (msg.supports_spk === false) {
    document.getElementById('speakerDiarization').checked = false;
    document.getElementById('speakerParams').classList.add('hidden');
  }
}

function handleSpkSaved(msg) {
  showSuccess('说话人库已保存：' + msg.name + ' (' + msg.num_speakers + '人)');
  document.getElementById('spkBankInfo').textContent = '已保存: ' + msg.name;
  document.getElementById('spkBankInfo').style.color = '#64ffda';
}

function handleSpkLoaded(msg) {
  showSuccess('说话人库已加载：' + msg.name + ' (' + msg.num_speakers + '人)');
  document.getElementById('spkBankInfo').textContent = '已加载: ' + msg.name;
  document.getElementById('spkBankInfo').style.color = '#64ffda';
  // 同步说话人数量滑块并持久化，同步到后端
  if (msg.preset_spk_num !== undefined) {
    document.getElementById('presetSpkNum').value = msg.preset_spk_num;
    document.getElementById('presetSpkNumValue').textContent = msg.preset_spk_num;
    saveConfigToLocal();
  }
}

function handleDeviceSwitched(msg) {
  const devInfo = msg.resolved === 'cuda:0' ? 'GPU' : 'CPU';
  showSuccess('已切换到 ' + devInfo + ' (' + msg.resolved + ')');
  checkDeviceStatus();
}

function handleDeviceSelected(msg) {
  const devName = msg.device_index !== null ? '设备 #' + msg.device_index : '自动检测';
  showSuccess('已切换音频设备：' + devName);
}

function handlePartialResult(msg) {
  // is_final: Pass1 定稿，只清空 Pass1 显示区域，不添加到 finalSentences
  // Pass2 结果或后端 fallback 消息会通过 type:"final" 持久化
  if (msg.is_final) {
    AppState.transcription.currentPartial = '';
    updatePartial();
    return;
  }
  const newText = msg.text || '';
  if (newText !== AppState.transcription.currentPartial) {
    AppState.transcription.currentPartial = newText;
    updatePartial();
  }
  let pass1Area = document.getElementById('pass1Area');
  if (pass1Area) pass1Area.style.display = '';
  if (msg.latency) {
    updateLatency(1, msg.latency);
  }
}

/**
 * 限制最大句子数，防止长时间转录内存无限增长。
 * 超过 MAX_SENTENCES 时裁剪旧句子并同步清理 DOM。
 * 优化：批量移除 DOM 节点，使用 Range API 减少 reflow。
 */
function _trimSentences() {
  const MAX_SENTENCES = 300;
  if (AppState.transcription.finalSentences.length > MAX_SENTENCES) {
    const removed = AppState.transcription.finalSentences.length - MAX_SENTENCES;
    AppState.transcription.finalSentences = AppState.transcription.finalSentences.slice(removed);
    AppState.transcription._renderedSentenceCount = Math.max(0, AppState.transcription._renderedSentenceCount - removed);
    const area = document.getElementById('transcriptionArea');
    if (area) {
      // 批量移除：先收集要移除的节点，再一次移除，减少 reflow
      const toRemove = area.children.length - MAX_SENTENCES;
      if (toRemove > 0) {
        const range = document.createRange();
        range.setStartBefore(area.firstChild);
        let node = area.firstChild;
        for (let i = 1; i < toRemove; i++) {
          node = node.nextSibling;
        }
        range.setEndAfter(node);
        range.deleteContents();
      }
    }
  }
}

function handleFinalResult(msg) {
  const isFallback = msg.source === 'pass1_fallback';
  // Pass2 精确结果到达时，替换所有连续的 _pass1_fallback 条目（去重）
  const hasContent = (msg.sentences && msg.sentences.length > 0) || msg.text;
  if (hasContent && !isFallback) {
    let fallbackCount = 0;
    while (AppState.transcription.finalSentences.length > 0 && AppState.transcription.finalSentences[AppState.transcription.finalSentences.length - 1]._pass1_fallback) {
      AppState.transcription.finalSentences.pop();
      fallbackCount++;
    }
    if (fallbackCount > 0) {
      AppState.transcription._renderedSentenceCount = Math.max(0, AppState.transcription._renderedSentenceCount - fallbackCount);
      const area = document.getElementById('transcriptionArea');
      if (area) {
        // 批量移除：使用 Range API 一次性移除多个节点，避免逐个 removeChild 导致多次 reflow
        const toRemove = Math.min(fallbackCount, area.children.length);
        if (toRemove > 0) {
          const range = document.createRange();
          let node = area.lastElementChild;
          // 从末尾向前找到第一个要移除的节点
          for (let i = 1; i < toRemove; i++) {
            node = node.previousElementSibling;
          }
          range.setStartBefore(node);
          range.setEndAfter(area.lastElementChild);
          range.deleteContents();
        }
      }
    }
  }
  if (msg.sentences && msg.sentences.length > 0) {
    // fallback 消息的句子标记 _pass1_fallback，以便后续 Pass2 结果替换
    let sentences = msg.sentences;
    if (isFallback) {
      sentences = msg.sentences.map(function(s) { s._pass1_fallback = true; return s; });
    }
    AppState.transcription.finalSentences = AppState.transcription.finalSentences.concat(sentences);
    _trimSentences();
  } else if (msg.text) {
    AppState.transcription.finalSentences.push({
      text: msg.text,
      start_ms: msg.start_ms || 0,
      end_ms: msg.end_ms || 0,
      spk: msg.spk !== undefined ? msg.spk : -1,
      _pass1_fallback: isFallback || undefined
    });
    _trimSentences();
  }
  renderTranscription();
  // Update sentence count
  let sentenceCountEl = document.getElementById('sentenceCount');
  if (sentenceCountEl) {
    sentenceCountEl.textContent = AppState.transcription.finalSentences.length + ' 句';
  }
  showEl('sentenceCount', 'show-inline');
  if (msg.latency) {
    updateLatency(2, msg.latency);
  }
}

function handleMessage(data) {
  let msg;
  try { msg = JSON.parse(data); } catch (e) { return; }

  // 停止确认
  if (AppState.transcription.waitingForStop && msg && msg.event === 'stopped') {
    AppState.transcription.waitingForStop = false;
  }

  // 事件分发
  const eventHandlers = {
    'capture_error': function(msg) { showError(msg.message || msg.error || '音频捕获错误'); },
    'error': handleError,
    'model_switched': handleModelSwitched,
    'model_error': function(msg) { showError(msg.message || '模型切换失败'); },
    'spk_error': function(msg) { showError(msg.message || '说话人分离错误'); },
    'spk_saved': handleSpkSaved,
    'spk_save_error': function(msg) { showError(msg.message || '保存说话人库失败'); },
    'spk_loaded': handleSpkLoaded,
    'spk_load_error': function(msg) { showError(msg.message || '加载说话人库失败'); },
    'device_switched': handleDeviceSwitched,
    'device_error': function(msg) { showError(msg.message || '设备切换失败'); },
    'device_selected': handleDeviceSelected,
    'device_select_error': function(msg) { showError(msg.message || '设备选择失败'); },
    'stopped': function() { cleanupAfterStop(); },
    'started': function() {
        console.log('[WS] 转录已启动');
    }
  };

  if (msg.event && eventHandlers[msg.event]) {
    eventHandlers[msg.event](msg);
    return;
  }

  // 类型分发
  if (msg.type === 'partial') {
    handlePartialResult(msg);
  } else if (msg.type === 'final') {
    handleFinalResult(msg);
  }
}

async function startTranscription() {
    if (AppState.transcription._isStarting) return;  // 防重入
    AppState.transcription._isStarting = true;
    hideError();
  const serverUrlEl = document.getElementById('serverUrl');
  if (!serverUrlEl) { AppState.transcription._isStarting = false; return; }
  const url = serverUrlEl.value.trim();
  const lang = document.getElementById('language').value;

  if (!url) {
    showError('请输入服务器地址', false);
    AppState.transcription._isStarting = false;
    return;
  }

  let wsUrl = url;

  try {
    await connectWebSocket(wsUrl);
  } catch (e) {
    showError('连接失败：' + (e.message || '无法连接到服务器'), true);
    AppState.transcription._isStarting = false;
    return;
  }

  const offlineModel = document.getElementById('offlineModel').value;
  AppState.connection.ws.send('MODEL_SWITCH:' + offlineModel);

  const device = document.getElementById('deviceSelect').value;
  AppState.connection.ws.send('DEVICE_SWITCH:' + device);

  const audioDevice = document.getElementById('audioDeviceSelect').value;
  AppState.connection.ws.send('DEVICE_SELECT:' + audioDevice);

  if (document.getElementById('speakerDiarization').checked) {
    AppState.connection.ws.send('SPK_ON');
  }

  if (lang && lang !== 'auto') {
    AppState.connection.ws.send('LANGUAGE:' + lang);
  }

  // 从 textarea 获取热词
  if (AppState.transcription.hotwords) {
    AppState.connection.ws.send('HOTWORDS:' + AppState.transcription.hotwords);
  }

  AppState.connection.ws.send('START');

  AppState.connection.ws.send('CAPTURE_START');
  setTranscribing(true);
  AppState.transcription._isStarting = false;
}

function stopTranscription() {
  if (AppState.connection.ws && AppState.connection.ws.readyState === WebSocket.OPEN) {
    AppState.connection.ws.send('CAPTURE_STOP');
  }

  if (AppState.connection.ws && AppState.connection.ws.readyState === WebSocket.OPEN) {
    AppState.connection.ws.send('STOP');
    AppState.transcription.waitingForStop = true;
    setTimeout(function() {
      if (AppState.transcription.waitingForStop) {
        AppState.transcription.waitingForStop = false;
        cleanupAfterStop();
      }
    }, 3000);
  } else {
    cleanupAfterStop();
  }
}

function cleanupAfterStop() {
  setConnected(false);
  setTranscribing(false);
  AppState.transcription.currentPartial = '';
  updatePartial();
  // 保留 Pass2 区域的精炼结果，停止后不清空
  // AppState.transcription.finalSentences 和 AppState.transcription._renderedSentenceCount 保持不变

  // Hide sentence count
  hideEl('sentenceCount');

  if (AppState.connection.ws) {
    try { AppState.connection.ws.close(); } catch(e) { console.warn('[WS] 关闭连接失败:', e); }
    AppState.connection.ws = null;
  }

  AppState.transcription.latencyHistory = {1: [], 2: []};
  hideEl('latencyPanel');
  document.getElementById('pass1Latency').textContent = '--';
  document.getElementById('pass1Rtf').textContent = 'RTF: --';
  document.getElementById('pass2Latency').textContent = '--';
  document.getElementById('pass2Rtf').textContent = 'RTF: --';
}

function retryConnect() {
  hideError();
  startTranscription();
}

// ===== 说话人库保存/加载 =====

function saveSpeakerBank() {
  const nameInput = document.getElementById('spkBankName');
  const name = nameInput ? nameInput.value.trim() : '';
  if (AppState.connection.ws && AppState.connection.ws.readyState === WebSocket.OPEN) {
    // WebSocket 连接中，通过 WebSocket 保存
    const payload = JSON.stringify({ name: name });
    AppState.connection.ws.send('SPK_SAVE:' + payload);
  } else {
    // WebSocket 已断开，通过 REST API 保存最后一次转录的说话人库
    const serverUrl = wsToHttp(document.getElementById('serverUrl').value);
    fetch(serverUrl + '/speaker-bank/save-last', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        showSuccess('说话人库已保存：' + data.name + ' (' + data.num_speakers + '人)');
        document.getElementById('spkBankInfo').textContent = '已保存: ' + data.name;
        document.getElementById('spkBankInfo').style.color = '#64ffda';
      } else {
        showError(data.error || '保存说话人库失败', false);
      }
    })
    .catch(function(e) { showError('保存失败: ' + e.message, false); });
  }
}

async function loadSpeakerBankList() {
  try {
    const serverUrl = wsToHttp(document.getElementById('serverUrl').value);
    const response = await fetch(serverUrl + '/speaker-banks');
    const data = await response.json();
    const banks = data.banks || [];
    if (banks.length === 0) {
      showError('没有已保存的说话人库', false);
      return;
    }
    // 显示选择下拉框
    const select = document.getElementById('spkBankSelect');
    if (!select) return;
    select.innerHTML = '<option value="">-- 选择说话人库 --</option>';
    for (let i = 0; i < banks.length; i++) {
      const b = banks[i];
      const opt = document.createElement('option');
      opt.value = b.id;
      opt.textContent = b.name + ' (' + b.num_speakers + '人, ' + (b.created_at || '未知') + ')';
      select.appendChild(opt);
    }
    document.getElementById('spkBankListArea').classList.remove('hidden');
  } catch (e) {
    showError('加载说话人库列表失败：' + e.message, false);
  }
}

function confirmLoadSpeakerBank() {
  const select = document.getElementById('spkBankSelect');
  if (!select || !select.value) return;
  const bankId = select.value;
  // 将 WebSocket URL 转换为 HTTP URL
  const serverUrl = wsToHttp(document.getElementById('serverUrl').value);

  if (AppState.connection.ws && AppState.connection.ws.readyState === WebSocket.OPEN) {
    // WebSocket 连接中，通过 WebSocket 加载（立即应用到当前 session）
    AppState.connection.ws.send('SPK_LOAD:' + bankId);
  } else {
    // WebSocket 未连接，通过 REST API 预加载（下次转录时自动应用）
    fetch(serverUrl + '/speaker-bank/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: bankId })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        showSuccess('说话人库已预加载：' + data.name + ' (' + data.num_speakers + '人)，开始转录时自动应用');
        document.getElementById('spkBankInfo').textContent = '已预加载: ' + data.name;
        document.getElementById('spkBankInfo').style.color = '#ffd54f';
        if (data.preset_spk_num !== undefined) {
          document.getElementById('presetSpkNum').value = data.preset_spk_num;
          document.getElementById('presetSpkNumValue').textContent = data.preset_spk_num;
          saveConfigToLocal();
        }
      } else {
        showError(data.error || '加载说话人库失败', false);
      }
    })
    .catch(function(e) { showError('加载失败: ' + e.message, false); });
  }
  document.getElementById('spkBankListArea').classList.add('hidden');
}

// ===== 复制/导出转录结果 =====

function formatTranscriptionText() {
  const lines = [];
  for (let i = 0; i < AppState.transcription.finalSentences.length; i++) {
    const s = AppState.transcription.finalSentences[i];
    const spk = s.spk >= 0 ? '[SPK' + s.spk + '] ' : '';
    const ts = s.start_ms !== undefined ? '[' + formatTime(s.start_ms) + '] ' : '';
    lines.push(ts + spk + s.text);
  }
  return lines.join('\n');
}

function copyTranscription() {
  if (AppState.transcription.finalSentences.length === 0) {
    showError('没有可复制的转录结果', false);
    return;
  }
  const text = formatTranscriptionText();
  navigator.clipboard.writeText(text).then(function() {
    showSuccess('已复制到剪贴板');
  }).catch(function(e) {
    showError('复制失败: ' + e.message, false);
  });
}

function exportTranscription() {
  if (AppState.transcription.finalSentences.length === 0) {
    showError('没有可导出的转录结果', false);
    return;
  }
  const text = formatTranscriptionText();
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const now = new Date();
  a.download = 'transcription_' + now.toISOString().slice(0, 19).replace(/[T:]/g, '-') + '.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showSuccess('已导出转录结果');
}
