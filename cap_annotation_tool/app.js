let tasks = null;
let annotations = null;
let currentIndex = 0;
let currentImage = new Image();
let filter = "all";
let boxes = [];
let selectedBoxId = null;
let drawing = null;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const frameList = document.getElementById("frameList");
const frameTitle = document.getElementById("frameTitle");
const frameSub = document.getElementById("frameSub");
const frameState = document.getElementById("frameState");
const notes = document.getElementById("notes");
const boxType = document.getElementById("boxType");
const boxState = document.getElementById("boxState");
const selectedInfo = document.getElementById("selectedInfo");
const autoPredictions = document.getElementById("autoPredictions");

const colors = {
  NORMAL: "#42d66b",
  DEFECT: "#ff4b3e",
  UNKNOWN: "#ffd43b",
  UNLABELED: "#39c6ff",
  PRESENT: "#42d66b",
  MISSING: "#ff4b3e",
  IGNORE: "#8b95a6",
  sample: "#39c6ff",
  fixed_region: "#b16bff",
  top: "#42d66b",
  bottom: "#42d66b",
  side_upper: "#ffd43b",
  side_lower: "#ffd43b",
  ignore: "#8b95a6",
};

async function loadJson(url, fallback) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status}`);
    return await res.json();
  } catch {
    return fallback;
  }
}

async function init() {
  tasks = await loadJson("/api/tasks", await loadJson("tasks.json", null));
  annotations = await loadJson("/api/annotations", await loadJson("annotations.json", { version: 1, frames: {} }));
  if (!tasks) {
    document.body.innerHTML = "<p style='padding:24px;color:white'>tasks.json 没有加载成功。</p>";
    return;
  }
  for (const frame of tasks.frames) {
    if (!annotations.frames[frame.id]) {
      annotations.frames[frame.id] = { frame_state: "UNLABELED", missing_parts: [], notes: "", objects: [] };
    }
  }
  document.getElementById("datasetMeta").textContent = `${tasks.frames.length} frames | right crop x>=${tasks.right_half_x}`;
  bindEvents();
  renderFrameList();
  await loadFrame(0);
}

function currentFrame() {
  return tasks.frames[currentIndex];
}

function currentAnno() {
  return annotations.frames[currentFrame().id];
}

function visibleFrames() {
  return tasks.frames
    .map((frame, index) => ({ frame, index }))
    .filter((item) => filter === "all" || item.frame.bucket === filter);
}

function renderFrameList() {
  frameList.innerHTML = "";
  for (const { frame, index } of visibleFrames()) {
    const anno = annotations.frames[frame.id] || {};
    const item = document.createElement("div");
    item.className = `frame-item${index === currentIndex ? " active" : ""}`;
    item.innerHTML = `
      <img src="${frame.filename}" alt="${frame.id}">
      <div>
        <div class="frame-name">${frame.id} | ${frame.time_sec.toFixed(2)}s</div>
        <div class="frame-meta">${frame.bucket}</div>
        <div class="frame-status">${anno.frame_state || "UNLABELED"} | ${(anno.objects || []).length} boxes</div>
      </div>
    `;
    item.addEventListener("click", () => loadFrame(index));
    frameList.appendChild(item);
  }
}

async function loadFrame(index) {
  saveFormToAnno();
  currentIndex = Math.max(0, Math.min(tasks.frames.length - 1, index));
  selectedBoxId = null;
  const frame = currentFrame();
  boxes = currentAnno().objects || [];
  currentImage = new Image();
  currentImage.onload = () => {
    canvas.width = frame.width;
    canvas.height = frame.height;
    fitCanvas();
    loadAnnoToForm();
    draw();
    renderFrameList();
    renderAutoPredictions();
  };
  currentImage.src = frame.filename;
}

function fitCanvas() {
  const wrap = canvas.parentElement.getBoundingClientRect();
  const scale = Math.min((wrap.width - 24) / canvas.width, (wrap.height - 24) / canvas.height, 1);
  canvas.style.width = `${Math.max(200, canvas.width * scale)}px`;
  canvas.style.height = `${Math.max(200, canvas.height * scale)}px`;
}

function loadAnnoToForm() {
  const anno = currentAnno();
  frameTitle.textContent = `${currentFrame().id}  t=${currentFrame().time_sec.toFixed(2)}s`;
  frameSub.textContent = `frame=${currentFrame().frame_idx} | 坐标保存为原视频坐标，当前图像 offset_x=${currentFrame().offset_x}`;
  frameState.value = anno.frame_state || "UNLABELED";
  notes.value = anno.notes || "";
  document.querySelectorAll(".checks input").forEach((input) => {
    input.checked = (anno.missing_parts || []).includes(input.value);
  });
  selectedInfo.textContent = "未选中框";
}

function saveFormToAnno() {
  if (!tasks || !annotations) return;
  const anno = currentAnno();
  anno.frame_state = frameState.value;
  anno.notes = notes.value;
  anno.missing_parts = [...document.querySelectorAll(".checks input:checked")].map((input) => input.value);
  anno.objects = boxes;
}

function imageBoxFromStored(box) {
  return { x: box.x - currentFrame().offset_x, y: box.y, w: box.w, h: box.h };
}

function storedBoxFromImage(rect) {
  return {
    x: Math.round(rect.x + currentFrame().offset_x),
    y: Math.round(rect.y),
    w: Math.round(rect.w),
    h: Math.round(rect.h),
  };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,0.10)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#39c6ff";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 8]);
  ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  ctx.setLineDash([]);
  for (const box of boxes) drawBox(box, box.id === selectedBoxId);
  if (drawing) {
    drawImageRect(drawing, colors[boxType.value], `${boxType.value} new`, true);
  }
}

function drawBox(box, selected) {
  const rect = imageBoxFromStored(box);
  const color = colors[box.state] || colors[box.type] || "#39c6ff";
  drawImageRect(rect, color, `${box.type} ${box.state || ""}`, selected);
}

function drawImageRect(rect, color, label, selected) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 4 : 2;
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  const text = label.trim();
  const width = Math.max(76, ctx.measureText(text).width + 12);
  ctx.fillRect(rect.x, Math.max(0, rect.y - 24), width, 22);
  ctx.fillStyle = color;
  ctx.font = "15px Microsoft YaHei, Segoe UI, Arial";
  ctx.fillText(text, rect.x + 6, Math.max(16, rect.y - 7));
  ctx.restore();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function hitTest(point) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const rect = imageBoxFromStored(boxes[i]);
    if (point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h) {
      return boxes[i];
    }
  }
  return null;
}

function selectBox(id) {
  selectedBoxId = id;
  const box = boxes.find((item) => item.id === id);
  if (box) {
    boxState.value = box.state || "UNLABELED";
    selectedInfo.textContent = `${box.type} | x=${box.x}, y=${box.y}, w=${box.w}, h=${box.h}`;
  } else {
    selectedInfo.textContent = "未选中框";
  }
  draw();
}

function addBox(rect, type = boxType.value, state = boxState.value) {
  if (Math.abs(rect.w) < 6 || Math.abs(rect.h) < 6) return;
  const norm = {
    x: Math.min(rect.x, rect.x + rect.w),
    y: Math.min(rect.y, rect.y + rect.h),
    w: Math.abs(rect.w),
    h: Math.abs(rect.h),
  };
  const stored = storedBoxFromImage(norm);
  const box = {
    id: `b${Date.now()}_${Math.floor(Math.random() * 1000)}`,
    type,
    state,
    ...stored,
  };
  boxes.push(box);
  selectBox(box.id);
}

function renderAutoPredictions() {
  const preds = currentFrame().auto_predictions || [];
  if (!preds.length) {
    autoPredictions.innerHTML = "<div class='prediction'>没有当前时间点的 AI 参考框。</div>";
    return;
  }
  autoPredictions.innerHTML = preds
    .map(
      (pred) => `
      <div class="prediction">
        <span class="tag">#${pred.sample_index}</span>
        <span class="tag">${pred.state}</span>
        <div>missing: ${pred.missing_side || "-"}</div>
        <div>x=${pred.sample_box.x}, y=${pred.sample_box.y}, w=${pred.sample_box.w}, h=${pred.sample_box.h}</div>
      </div>`
    )
    .join("");
}

function useAutoBoxes() {
  const preds = currentFrame().auto_predictions || [];
  for (const pred of preds) {
    boxes.push({
      id: `auto_${Date.now()}_${pred.sample_index}`,
      type: "sample",
      state: pred.state === "DEFECT" ? "MISSING" : pred.state === "UNKNOWN" ? "UNKNOWN" : "PRESENT",
      ...pred.sample_box,
      source: "auto_prediction",
    });
  }
  draw();
}

function addFixedRegion() {
  boxes.push({
    id: `fixed_${Date.now()}`,
    type: "fixed_region",
    state: "UNLABELED",
    x: currentFrame().offset_x,
    y: 0,
    w: currentFrame().width,
    h: currentFrame().height,
  });
  draw();
}

async function saveAnnotations() {
  saveFormToAnno();
  try {
    const res = await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotations),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    alert("已保存到 annotations.json");
  } catch {
    download("annotations.json", JSON.stringify(annotations, null, 2), "application/json");
  }
  renderFrameList();
}

function exportCsv() {
  saveFormToAnno();
  const lines = ["frame_id,time_sec,frame_state,missing_parts,object_id,type,state,x,y,w,h,notes"];
  for (const frame of tasks.frames) {
    const anno = annotations.frames[frame.id] || {};
    const objects = anno.objects && anno.objects.length ? anno.objects : [{ id: "", type: "", state: "", x: "", y: "", w: "", h: "" }];
    for (const obj of objects) {
      lines.push(
        [
          frame.id,
          frame.time_sec,
          anno.frame_state || "UNLABELED",
          (anno.missing_parts || []).join("+"),
          obj.id || "",
          obj.type || "",
          obj.state || "",
          obj.x ?? "",
          obj.y ?? "",
          obj.w ?? "",
          obj.h ?? "",
          JSON.stringify(anno.notes || ""),
        ].join(",")
      );
    }
  }
  download("cap_annotations.csv", lines.join("\n"), "text/csv");
}

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  window.addEventListener("resize", () => {
    fitCanvas();
    draw();
  });
  canvas.addEventListener("mousedown", (event) => {
    const point = canvasPoint(event);
    const hit = hitTest(point);
    if (hit) {
      selectBox(hit.id);
      return;
    }
    selectedBoxId = null;
    drawing = { x: point.x, y: point.y, w: 0, h: 0 };
  });
  canvas.addEventListener("mousemove", (event) => {
    if (!drawing) return;
    const point = canvasPoint(event);
    drawing.w = point.x - drawing.x;
    drawing.h = point.y - drawing.y;
    draw();
  });
  canvas.addEventListener("mouseup", () => {
    if (drawing) addBox(drawing);
    drawing = null;
    draw();
  });
  document.querySelectorAll(".seg").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      filter = button.dataset.filter;
      renderFrameList();
    });
  });
  document.querySelectorAll(".state-btn").forEach((button) => {
    button.addEventListener("click", () => {
      frameState.value = button.dataset.state;
      saveFormToAnno();
      renderFrameList();
    });
  });
  frameState.addEventListener("change", () => {
    saveFormToAnno();
    renderFrameList();
  });
  notes.addEventListener("input", saveFormToAnno);
  document.querySelectorAll(".checks input").forEach((input) => input.addEventListener("change", saveFormToAnno));
  boxState.addEventListener("change", () => {
    const box = boxes.find((item) => item.id === selectedBoxId);
    if (box) {
      box.state = boxState.value;
      draw();
    }
  });
  document.getElementById("prevBtn").addEventListener("click", () => loadFrame(currentIndex - 1));
  document.getElementById("nextBtn").addEventListener("click", () => loadFrame(currentIndex + 1));
  document.getElementById("copyPrevBtn").addEventListener("click", () => {
    if (currentIndex <= 0) return;
    boxes = JSON.parse(JSON.stringify(annotations.frames[tasks.frames[currentIndex - 1].id].objects || []));
    draw();
  });
  document.getElementById("saveBtn").addEventListener("click", saveAnnotations);
  document.getElementById("exportJsonBtn").addEventListener("click", () => {
    saveFormToAnno();
    download("cap_annotations.json", JSON.stringify(annotations, null, 2), "application/json");
  });
  document.getElementById("exportCsvBtn").addEventListener("click", exportCsv);
  document.getElementById("useAutoBtn").addEventListener("click", useAutoBoxes);
  document.getElementById("addFixedRegionBtn").addEventListener("click", addFixedRegion);
  document.getElementById("deleteBoxBtn").addEventListener("click", () => {
    boxes = boxes.filter((item) => item.id !== selectedBoxId);
    selectedBoxId = null;
    draw();
  });
  document.getElementById("clearFrameBtn").addEventListener("click", () => {
    if (confirm("清空当前帧所有框？")) {
      boxes = [];
      selectedBoxId = null;
      draw();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "TEXTAREA" || event.target.tagName === "SELECT") return;
    if (event.key.toLowerCase() === "a") loadFrame(currentIndex - 1);
    if (event.key.toLowerCase() === "d") loadFrame(currentIndex + 1);
    if (event.key === "1") frameState.value = "NORMAL";
    if (event.key === "2") frameState.value = "DEFECT";
    if (event.key === "3") frameState.value = "UNKNOWN";
    if (["1", "2", "3"].includes(event.key)) {
      saveFormToAnno();
      renderFrameList();
    }
    if (event.key === "Delete" && selectedBoxId) {
      boxes = boxes.filter((item) => item.id !== selectedBoxId);
      selectedBoxId = null;
      draw();
    }
  });
}

init();
