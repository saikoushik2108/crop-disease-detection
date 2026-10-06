import React, { useState } from 'react';
import { Upload, MapPin, Thermometer, Leaf, Activity, AlertCircle, CheckCircle2, Eye, Zap, Info, Droplets, Wind } from 'lucide-react';
import { api } from './services/api';

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [variety, setVariety] = useState('ADT45');
  const [age, setAge] = useState(65);
  const [location, setLocation] = useState({ lat: null, lon: null, city: null });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showXAI, setShowXAI] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleGetLocation = () => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          city: "Detecting..."
        });
      },
      (err) => {
        setError("Unable to retrieve your location. Please enter manually.");
      }
    );
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please upload a leaf image");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setShowXAI(false);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('variety', variety);
    formData.append('age', age);
    if (location.lat) formData.append('lat', location.lat);
    if (location.lon) formData.append('lon', location.lon);

    try {
      const data = await api.predictDisease(formData);
      setResult(data);
    } catch (err) {
      setError("Prediction failed. Please check if the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-4 md:p-8">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex items-center gap-4">
          <div className="bg-green-500 p-3 rounded-2xl shadow-lg shadow-green-500/20">
            <Leaf className="text-white w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Paddy Doctor <span className="text-green-500">Multimodal AI</span></h1>
            <p className="text-slate-400 text-sm">In-Field Crop Disease Diagnostics (Vision + Cultivar + Phenology)</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full text-green-400 text-xs font-medium">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            ResNet18 Multimodal Fusion Live
          </div>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-lg border border-slate-700 transition-all">
            API Docs (Swagger) →
          </a>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Panel: Input */}
        <div className="lg:col-span-5 space-y-6">
          <section className="bg-slate-900/50 p-6 rounded-3xl shadow-xl border border-slate-800 backdrop-blur-sm">
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 text-white">
              <Upload className="w-5 h-5 text-green-500" />
              Leaf Image & Agronomic Metadata
            </h2>

            <div className="space-y-6">
              {/* Upload Area */}
              <div className="relative group cursor-pointer border-2 border-dashed border-slate-700 rounded-2xl p-6 transition-all hover:border-green-500 bg-slate-800/40 text-center">
                <input
                  type="file"
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  onChange={handleFileChange}
                  accept="image/*"
                />
                {preview ? (
                  <div className="relative">
                    <img src={preview} alt="Preview" className="w-full h-64 object-cover rounded-xl shadow-lg" />
                    <div className="absolute bottom-2 right-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded-md">Preview</div>
                  </div>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center text-slate-500">
                    <div className="bg-slate-800 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                      <Upload className="w-10 h-10 text-slate-400" />
                    </div>
                    <p className="font-medium text-slate-300">Click or drag & drop leaf photo</p>
                    <p className="text-xs mt-1">Supports JPEG / PNG field captures (1080p, 480p)</p>
                  </div>
                )}
              </div>

              {/* Presets */}
              <div className="flex flex-wrap gap-2">
                <span className="text-[10px] font-bold text-slate-500 uppercase w-full mb-1">Preset Samples</span>
                {['Blast (ADT45)', 'Panicle Blight (72d)', 'Dead Heart (Ponni)', 'Healthy Leaf (50d)'].map(preset => (
                  <button key={preset} className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs rounded-full border border-slate-700 transition-all">
                    {preset}
                  </button>
                ))}
              </div>

              {/* Metadata Inputs */}
              <div className="grid grid-cols-1 gap-6 pt-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-400 flex items-center gap-2">
                    <Leaf className="w-4 h-4 text-green-500" /> Paddy Cultivar / Variety
                  </label>
                  <select
                    value={variety}
                    onChange={(e) => setVariety(e.target.value)}
                    className="w-full p-3 rounded-xl border border-slate-700 bg-slate-800 text-white focus:ring-2 focus:ring-green-500 outline-none transition-all"
                  >
                    <option value="ADT45">ADT45 (High Yielding Tamil Nadu Short-Duration)</option>
                    <option value="IR64">IR64 (International Standard)</option>
                    <option value="Sonalika">Sonalika (Drought Resistant)</option>
                    <option value="Other">Other Variety</option>
                  </select>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium text-slate-400 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-green-500" /> Crop Age from Sowing (Days)
                    </label>
                    <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs font-bold rounded-md">{age} days</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="180"
                    value={age}
                    onChange={(e) => setAge(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 px-1">
                    <span>Seedling</span>
                    <span>Vegetative</span>
                    <span>Reproductive</span>
                    <span>Ripening</span>
                  </div>
                  <p className="text-[11px] text-blue-400 italic">Phenological Phase: Flowering / Heading Stage</p>
                </div>

                <div className="p-4 bg-slate-800/50 rounded-2xl border border-slate-700">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-slate-300 flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-green-500" /> Location Context
                    </span>
                    <button
                      onClick={handleGetLocation}
                      className="text-xs text-green-400 hover:text-green-300 font-bold underline"
                    >
                      Detect Now
                    </button>
                  </div>
                  {location.lat ? (
                    <div className="text-xs text-slate-400 flex items-center gap-2">
                      <CheckCircle2 className="w-3 h-3 text-green-500" />
                      GPS Active: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500 italic">Location not provided. Will use global baselines.</div>
                  )}
                </div>
              </div>
            </div>
          </section>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-4 bg-green-600 text-white rounded-2xl font-bold text-lg hover:bg-green-500 transition-all disabled:bg-slate-700 disabled:text-slate-500 shadow-lg shadow-green-600/20 flex items-center justify-center gap-3"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                Analyzing Multimodal Data...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Run Multimodal Diagnosis
              </>
            )}
          </button>

          {error && (
            <div className="p-4 bg-red-500/10 text-red-400 rounded-2xl border border-red-500/20 flex items-center gap-2 text-sm">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}
        </div>

        {/* Right Panel: Results */}
        <div className="lg:col-span-7 space-y-6">
          {!result && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 p-12 border-2 border-dashed border-slate-800 rounded-3xl bg-slate-900/30">
              <div className="bg-slate-800 p-6 rounded-full mb-6 opacity-20">
                <Activity className="w-16 h-16" />
              </div>
              <h3 className="text-xl font-semibold text-slate-300 mb-2">Awaiting Leaf Sample</h3>
              <p className="text-center max-w-xs">Upload a photograph and click "Run Multimodal Diagnosis" to generate findings.</p>
            </div>
          )}

          {loading && (
            <div className="h-full flex flex-col items-center justify-center p-12 bg-slate-900/50 rounded-3xl border border-slate-800">
              <div className="relative mb-6">
                <div className="w-20 h-20 border-4 border-green-500/20 border-t-green-500 rounded-full animate-spin"></div>
                <Leaf className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-green-500 w-8 h-8" />
              </div>
              <p className="text-slate-400 font-medium animate-pulse">Processing Fusion Network...</p>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
              {/* Primary Prediction */}
              <div className="bg-slate-900 p-8 rounded-3xl shadow-2xl border border-slate-800 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-20 group-hover:opacity-100 transition-opacity">
                   <Activity className="text-green-500 w-8 h-8" />
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest mb-4">Diagnostic Findings</h3>
                <div className="flex items-baseline gap-4 mb la-8">
                  <span className="text-5xl font-black text-white tracking-tight">{result.prediction.class}</span>
                  <span className="text-3xl font-bold text-green-500">{(result.prediction.confidence * 100).toFixed(1)}%</span>
                </div>

                <div className="mt-8 space-y-4">
                  <p className="text-sm font-medium text-slate-400 flex items-center gap-2">
                    <Info className="w-4 h-4" /> Top Differential Diagnoses
                  </p>
                  <div className="grid grid-cols-1 gap-2">
                    {result.top_predictions.map((p, i) => (
                      <div key={i} className={`flex items-center justify-between p-4 rounded-2xl transition-all ${i === 0 ? 'bg-green-500/10 border border-green-500/30' : 'bg-slate-800/50 border border-slate-700'}`}>
                        <span className={`text-sm font-semibold ${i === 0 ? 'text-green-400' : 'text-slate-300'}`}>{p.class}</span>
                        <span className="text-sm font-mono text-slate-400">{(p.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Explainability View */}
              <div className="bg-slate-900 p-6 rounded-3xl shadow-xl border border-slate-800">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-sm font-bold text-slate-400 uppercase flex items-center gap-2">
                    <Eye className="w-4 h-4 text-green-500" /> Spatial Influence (XAI)
                  </h4>
                  <button
                    onClick={() => setShowXAI(!showXAI)}
                    className="px-3 py-1 bg-green-500/10 text-green-400 text-xs font-bold rounded-full hover:bg-green-500/20 transition-all uppercase tracking-tighter"
                  >
                    {showXAI ? "Hide" : "Show"} Analysis
                  </button>
                </div>
                {showXAI && result.explainability?.status === 'available' && (
                  <div className="flex flex-col items-center gap-4 animate-in zoom-in-95 duration-300">
                    <div className="relative p-2 bg-slate-800 rounded-2xl border border-slate-700">
                      <img
                        src={result.explainability.overlay_b64}
                        alt="Grad-CAM Overlay"
                        className="w-full max-w-md rounded-xl shadow-2xl"
                      />
                    </div>
                    <p className="text-xs text-slate-500 text-center italic max-w-md">
                      Grad-CAM Visualization: Red regions highlight morphological features that strongly influenced the {result.prediction.class} prediction.
                    </p>
                  </div>
                )}
                {showXAI && result.explainability?.status === 'unavailable' && (
                  <div className="text-sm text-slate-500 italic text-center py-8">
                    Explainability data currently unavailable.
                  </div>
                )}
              </div>

              {/* Multi-Context Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Agronomic */}
                <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-4 flex items-center gap-2">
                    <Leaf className="w-3 h-3" /> Agronomic
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-slate-400">Variety</span>
                      <span className="text-xs font-bold text-white">{result.input_context.variety}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-slate-400">Age</span>
                      <span className="text-xs font-bold text-white">{result.input_context.age}d</span>
                    </div>
                  </div>
                </div>

                {/* Location */}
                <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-4 flex items-center gap-2">
                    <MapPin className="w-3 h-3" /> Location
                  </h4>
                  {result.location_context.status === 'available' ? (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-400">City</span>
                        <span className="text-xs font-bold text-white truncate ml-2">{result.location_context.city}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-400">Region</span>
                        <span className="text-xs font-bold text-white truncate ml-2">{result.location_context.region}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-slate-600 italic">Not available</div>
                  )}
                </div>

                {/* Environment */}
                <div className="bg-slate-900 p-5 rounded-3xl border border-slate-800">
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-4 flex items-center gap-2">
                    <Thermometer className="w-3 h-3" /> Environment
                  </h4>
                  {result.environment_context.status === 'available' ? (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-400">Temp</span>
                        <span className="text-xs font-bold text-white">{result.environment_context.temperature}°C</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-400">Humid</span>
                        <span className="text-xs font-bold text-white">{result.environment_context.humidity}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-400">Sky</span>
                        <span className="text-xs font-bold text-white truncate ml-2">{result.environment_context.condition}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-slate-600 italic">Not available</div>
                  )}
                </div>
              </div>

              {/* Scientific Disclaimer */}
              <div className="p-4 bg-blue-500/10 rounded-2xl border border-blue-500/20 text-blue-400 text-[11px] leading-relaxed flex gap-3">
                <Info className="w-4 h-4 shrink-0 mt-0.5" />
                <p>
                  <strong className="text-blue-300">Scientific Note:</strong> The environmental and location data displayed above are retrieved via real-time API and serve as contextual information. They are not used as trained input features for the core multimodal neural network.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
