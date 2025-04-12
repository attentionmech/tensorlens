import {
  Engine,
  Scene,
  ArcRotateCamera,
  Vector3,
  Color3,
  ShaderMaterial,
  PointsCloudSystem,
  Effect
} from "@babylonjs/core";
import "@babylonjs/loaders";

let spsMesh = null;
let currentTensorKey = null;
let refreshInterval = null;
let isFetching = false;

async function fetchTensorList() {
  const res = await fetch("/api/list_tensors");
  return await res.json();
}

async function fetchMaxPoints() {
  const res = await fetch("/api/config");
  const config = await res.json();
  return config.max_points || 10000; // Default to 10,000 if max_points is not set
}

async function fetchTensorByKey(key, code = "", maxPoints = 10000) {
  const url = `/api/get_tensor?tensor_key=${encodeURIComponent(key)}&code=${encodeURIComponent(code)}`;
  const res = await fetch(url);
  const tensor = await res.json();

  // Apply downsampling
  tensor.data = downsampleTensor(tensor.data, tensor.dims, maxPoints);
  return tensor;
}

function neonColorMap(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const r = clamped;
  const g = (1 - clamped) * 0.8;
  const b = clamped * 0.9 + (1 - clamped) * 1;
  return new Color3(r, g, b);
}

function clearRenderedMesh() {
  if (spsMesh) {
    spsMesh.dispose();
    spsMesh = null;
  }
}

// Custom shaders
Effect.ShadersStore["customVertexShader"] = `
  precision highp float;
  attribute vec3 position;
  attribute vec4 color;
  uniform mat4 worldViewProjection;
  uniform float pointSize;
  varying vec4 vColor;

  void main() {
    gl_Position = worldViewProjection * vec4(position, 1.0);
    gl_PointSize = pointSize;
    vColor = color;
  }
`;

Effect.ShadersStore["customFragmentShader"] = `
  precision highp float;
  varying vec4 vColor;

  void main() {
    gl_FragColor = vColor;
  }
`;

function renderTensorWithPCS(tensor, scene, offset) {
  const dims = tensor.dims;
  const data = tensor.data;

  const spacing = 0.7;
  const positions = [];

  if (dims === 1) {
    data.forEach((v, i) => {
      positions.push({ pos: new Vector3(offset.x + i * spacing, offset.y, offset.z), val: v });
    });
  } else if (dims === 2) {
    data.forEach((row, y) => {
      row.forEach((v, x) => {
        positions.push({ pos: new Vector3(offset.x + x * spacing, offset.y, offset.z + y * spacing), val: v });
      });
    });
  } else if (dims === 3) {
    data.forEach((plane, z) => {
      plane.forEach((row, y) => {
        row.forEach((v, x) => {
          positions.push({
            pos: new Vector3(
              offset.x + x * spacing,
              offset.y + y * spacing,
              offset.z + z * spacing
            ),
            val: v
          });
        });
      });
    });
  } else {
    console.log(`Unsupported tensor dimension: ${dims}`);
    return;
  }

  console.log("Rendering", positions.length, "points");

  const center = positions.reduce((acc, p) => acc.add(p.pos), new Vector3(0, 0, 0)).scale(1 / positions.length);
  scene.activeCamera.setTarget(center);

  const pcs = new PointsCloudSystem("pcs", 1, scene);

  pcs.addPoints(positions.length, (particle, i) => {
    particle.position = positions[i].pos;
    const color = neonColorMap(positions[i].val);
    particle.color = new Color3(color.r, color.g, color.b);
  });

  pcs.buildMeshAsync().then(mesh => {
    spsMesh = mesh;

    const shaderMat = new ShaderMaterial("pointShader", scene, {
      vertex: "custom",
      fragment: "custom",
    }, {
      attributes: ["position", "color"],
      uniforms: ["worldViewProjection", "pointSize"],
    });

    shaderMat.setFloat("pointSize", 6.0);
    shaderMat.backFaceCulling = false;
    shaderMat.pointsCloud = true;
    shaderMat.disableLighting = true;
    shaderMat.emissiveColor = new Color3(1, 1, 1);

    mesh.material = shaderMat;
    mesh.alwaysSelectAsActiveMesh = true;
  });
}

// Show downsampling message
function showDownsamplingMessage() {
  const messageElement = document.getElementById("downsampleMessage");
  messageElement.style.display = "block"; // Show the message

  // Hide the message after 3 seconds
  setTimeout(() => {
    messageElement.style.display = "none";
  }, 3000); // 3000ms = 3 seconds
}

// Update the downsampleTensor function to call showDownsamplingMessage when downsampling is triggered
function downsampleTensor(data, dims, maxPoints) {
  let downsampledData = data;

  if (dims === 1) {
    if (data.length > maxPoints) {
      downsampledData = sample1D(data, maxPoints);
      showDownsamplingMessage();  // Show the message when downsampling occurs
    }
  } else if (dims === 2) {
    const total = data.reduce((sum, row) => sum + row.length, 0);
    if (total > maxPoints) {
      downsampledData = sample2D(data, maxPoints);
      showDownsamplingMessage();  // Show the message when downsampling occurs
    }
  } else if (dims === 3) {
    const total = data.reduce((sum, plane) =>
      sum + plane.reduce((rowSum, row) => rowSum + row.length, 0), 0);
    if (total > maxPoints) {
      downsampledData = sample3D(data, maxPoints);
      showDownsamplingMessage();  // Show the message when downsampling occurs
    }
  } else {
    console.warn("Unsupported dims for downsampling:", dims);
  }

  return downsampledData;
}

function sample1D(data, maxPoints) {
  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, idx) => idx % step === 0);
}

function sample2D(data, maxPoints) {
  const flat = [];
  data.forEach(row => flat.push(...row));
  const step = Math.ceil(flat.length / maxPoints);
  const sampled = flat.filter((_, idx) => idx % step === 0);

  // Rebuild into 2D shape (as square as possible)
  const size = Math.ceil(Math.sqrt(sampled.length));
  const out = [];
  for (let i = 0; i < sampled.length; i += size) {
    out.push(sampled.slice(i, i + size));
  }
  return out;
}

function sample3D(data, maxPoints) {
  const flat = [];
  data.forEach(plane => plane.forEach(row => flat.push(...row)));
  const step = Math.ceil(flat.length / maxPoints);
  const sampled = flat.filter((_, idx) => idx % step === 0);

  // Rebuild into 3D cube shape
  const cubeSize = Math.ceil(Math.cbrt(sampled.length));
  const out = [];
  for (let z = 0; z < cubeSize; z++) {
    const plane = [];
    for (let y = 0; y < cubeSize; y++) {
      const row = [];
      for (let x = 0; x < cubeSize; x++) {
        const i = z * cubeSize * cubeSize + y * cubeSize + x;
        if (i < sampled.length) {
          row.push(sampled[i]);
        }
      }
      if (row.length > 0) plane.push(row);
    }
    if (plane.length > 0) out.push(plane);
  }
  return out;
}

document.addEventListener("DOMContentLoaded", async () => {
  const maxPoints = await fetchMaxPoints(); // ← Fetch it once here

  const canvas = document.getElementById("renderCanvas");
  const engine = new Engine(canvas, true);
  const scene = new Scene(engine);
  scene.clearColor = new Color3(0, 0, 0);

  const camera = new ArcRotateCamera("camera", Math.PI / 4, Math.PI / 4, 50, Vector3.Zero(), scene);
  camera.minZ = 0.01;
  camera.maxZ = 10000;
  camera.attachControl(canvas, true);
  camera.wheelDeltaPercentage = 0.01;

  const selector = document.getElementById("tensorSelector");
  const pythonCodeInput = document.getElementById("pythonCodeInput");
  const infoDiv = document.getElementById("tensorInfo");

  const tensorList = await fetchTensorList();
  const tensorInfoMap = {};

  selector.innerHTML = '<option value="">Select a tensor</option>';
  tensorList.available_tensors.forEach(tensor => {
    const option = document.createElement("option");
    option.value = tensor.key;
    option.textContent = tensor.name || tensor.key;
    selector.appendChild(option);

    tensorInfoMap[tensor.key] = {
      name: tensor.name || tensor.key,
      shape: tensor.shape || "Unknown"
    };
  });

  selector.addEventListener("change", async (event) => {
    const key = event.target.value;
    if (!key) return;

    currentTensorKey = key;
    const tensorMeta = tensorInfoMap[key];
    infoDiv.textContent = `Tensor: ${tensorMeta.name} | Shape: ${tensorMeta.shape}`;

    if (refreshInterval) clearInterval(refreshInterval);

    const fetchAndRender = async () => {
      if (isFetching) return;
      isFetching = true;

      const pythonCode = pythonCodeInput.value;

      try {
        const tensor = await fetchTensorByKey(currentTensorKey, pythonCode, maxPoints); // ← pass it here
        clearRenderedMesh();
        renderTensorWithPCS(tensor, scene, new Vector3(0, 0, 0));
      } catch (err) {
        console.error("Failed to fetch tensor:", err);
      }

      isFetching = false;
    };

    await fetchAndRender();
    refreshInterval = setInterval(fetchAndRender, 500);
  });

  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => engine.resize());
});
