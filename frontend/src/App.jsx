import React, { useState } from 'react';
import { Upload, MapPin, Thermometer, Leaf, Activity, AlertCircle, CheckCircle2, Eye } from 'lucide-react';
import { api } from './services/api';

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [variety, setVariety] = useState('ADT45');
  const [age, setAge] = useState(45);
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
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-4 md:p-8">
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-8 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Leaf className="text-green-600 w-8 h-8" />
          <h1 className="text-3xl font-bold text-slate-800">Paddy Disease AI</h1>
        </div>
        <p className="text-slate-500">Multimodal Multi-Context Disease Classification System</p>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Input Section */}
        <div className="lg:col-span-5 space-y-6">
          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-green-600" />
              1. Upload Leaf Image
            </h2>
            <div className="relative group cursor-pointer border-2 border-dashed border-slate-300 rounded-xl p-4 transition-all hover:border-green-500 bg-slate-50">
              <input
                type="file"
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={handleFileChange}
                accept="image/*"
              />
              {preview ? (
                <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded-lg" />
              ) : (
                <div className="h-48 flex flex-col items-center justify-center text-slate-400">
                  <Upload className="w-10 h-10 mb-2" />
                  <p>Click or drag image here</p>
                </div>
              )}
            </div>
          </section>

          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-600" />
              2. Agronomic Context
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-600">Paddy Variety</label>
                <select
                  value={variety}
                  onChange={(e) => setVariety(e.target.value)}
                  className="w-full p-2 rounded-lg border border-slate-300 bg-white"
                >
                  <option value="ADT45">ADT45</option>
                  <option value="IR64">IR64</option>
                  <option value="Sonalika">Sonalika</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-600">Plant Age (Days)</label>
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(parseInt(e.target.value))}
                  className="w-full p-2 rounded-lg border border-slate-300"
                />
              </div>
            </div>
          </section>

          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <MapPin className="w-5 h-5 text-green-600" />
              3. Location Context
            </h2>
            <div className="flex gap-2">
              <button
                onClick={handleGetLocation}
                className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center justify-center gap-2"
              >
                <MapPin className="w-4 h-4" />
                Use My Location
              </button>
            </div>
            {location.lat && (
              <div className="mt-3 p-3 bg-green-50 rounded-lg text-xs text-green-700 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Location acquired: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
              </div>
            )}
          </section>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-4 bg-slate-900 text-white rounded-2xl font-bold text-lg hover:bg-slate-800 transition-all disabled:bg-slate-400 shadow-lg"
          >
            {loading ? "Analyzing Multimodal Data..." : "Run Prediction"}
          </button>

          {error && (
            <div className="p-4 bg-red-50 text-red-600 rounded-xl border border-red-200 flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}
        </div>

        {/* Output Section */}
        <div className="lg:col-span-7 space-y-6">
          {!result && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 p-12 border-2 border-dashed border-slate-200 rounded-3xl bg-white">
              <Leaf className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-lg">Upload image and context to see results</p>
            </div>
          )}

          {loading && (
            <div className="h-full flex flex-col items-center justify-center p-12 bg-white rounded-3xl shadow-sm border border-slate-200">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mb-4"></div>
              <p className="text-slate-500 font-medium">Processing fusion network...</p>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Primary Prediction */}
              <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4">
                   <Activity className="text-green-500 w-6 h-6" />
                </div>
                <h3 className="text-slate-500 text-sm font-medium uppercase tracking-wider mb-2">Core Model Prediction</h3>
                <div className="flex items-baseline gap-3 mb-6">
                  <span className="text-4xl font-bold text-slate-900">{result.prediction.class}</span>
                  <span className="text-2xl font-semibold text-green-600">{(result.prediction.confidence * 100).toFixed(1)}%</span>
                </div>

                <div className="space-y-3">
                  <p className="text-sm font-medium text-slate-600">Top 3 Candidates:</p>
                  {result.top_predictions.map((p, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                      <span className="text-sm font-medium">{p.class}</span>
                      <span className="text-sm text-slate-500">{(p.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Explainability View */}
              <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-sm font-bold text-slate-400 uppercase flex items-center gap-2">
                    <Eye className="w-4 h-4" /> Visual Explanation
                  </h4>
                  <button
                    onClick={() => setShowXAI(!showXAI)}
                    className="text-xs font-bold text-green-600 hover:text-green-700 uppercase tracking-tighter"
                  >
                    {showXAI ? "Hide" : "Show"} Analysis
                  </button>
                </div>
                {showXAI && result.explainability?.status === 'available' && (
                  <div className="flex flex-col items-center gap-4 animate-in fade-in slide-in-from-top-2">
                    <img
                      src={result.explainability.overlay_b64}
                      alt="Grad-CAM Overlay"
                      className="w-full max-w-md rounded-xl shadow-inner border border-slate-100"
                    />
                    <p className="text-xs text-slate-500 text-center italic">
                      Heatmap overlay: Red regions indicate high influence on the {result.prediction.class} prediction.
                    </p>
                  </div>
                )}
                {showXAI && result.explainability?.status === 'unavailable' && (
                  <div className="text-sm text-slate-400 italic text-center py-4">
                    Explainability data currently unavailable.
                  </div>
                )}
              </div>

              {/* Multi-Context Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Agronomic Context */}
                <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
                  <h4 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                    <Leaf className="w-4 h-4" /> Agronomic Context
                  </h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-500">Variety</span>
                      <span className="text-sm font-semibold">{result.input_context.variety}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-slate-500">Plant Age</span>
                      <span className="text-sm font-semibold">{result.input_context.age} days</span>
                    </div>
                  </div>
                </div>

                {/* Location Context */}
                <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
                  <h4 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                    <MapPin className="w-4 h-4" /> Location Context
                  </h4>
                  {result.location_context.status === 'available' ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">City</span>
                        <span className="text-sm font-semibold">{result.location_context.city}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Region</span>
                        <span className="text-sm font-semibold">{result.location_context.region}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Country</span>
                        <span className="text-sm font-semibold">{result.location_context.country}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-400 italic">Location context unavailable</div>
                  )}
                </div>

                {/* Environmental Context */}
                <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
                  <h4 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                    <Thermometer className="w-4 h-4" /> Environmental Context
                  </h4>
                  {result.environment_context.status === 'available' ? (
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Temperature</span>
                        <span className="text-sm font-semibold">{result.environment_context.temperature}°C</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Humidity</span>
                        <span className="text-sm font-semibold">{result.environment_context.humidity}%</span>
                      </span>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Condition</span>
                        <span className="text-sm font-semibold">{result.environment_context.condition}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-400 italic">Weather context unavailable</div>
                  )}
                </div>

                {/* Context Status */}
                <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
                  <h4 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Modality Status
                  </h>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.context_status).map(([key, status]) => (
                      <span key={key} className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase ${
                        status === 'available' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'
                      }`}>
                        {key}: {status}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Scientific Disclaimer */}
              <div className="p-4 bg-blue-50 rounded-2xl border border-blue-100 text-blue-700 text-xs leading-relaxed">
                <strong>Scientific Note:</strong> The environmental and location data displayed above are retrieved in real-time and are not used as training features for the core disease classifier.
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
