import React, { useState, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { FlyToInterpolator } from '@deck.gl/core';
import { ScatterplotLayer, PolygonLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { getVisualizationData, predictDebris, fetchPatchInference } from '../services/api';
import { Upload, Target } from 'lucide-react';

const INITIAL_VIEW_STATE = {
  longitude: -122.4194,
  latitude: 37.7749,
  zoom: 11,
  pitch: 40,
  bearing: 0
};

// Using ESRI World Imagery pure satellite base map
const MAP_STYLE = {
  version: 8,
  sources: {
    satellite: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      ],
      tileSize: 256,
      attribution: "Tiles &copy; Esri"
    }
  },
  layers: [
    {
      id: "background",
      type: "background",
      paint: { "background-color": "#021019" }
    },
    {
      id: "satellite-layer",
      type: "raster",
      source: "satellite",
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const Visualization = () => {
  const [data, setData] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [patchLoading, setPatchLoading] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [draftBBox, setDraftBBox] = useState(null);
  const [dateRange, setDateRange] = useState("last_3_days");
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const handleRunPatchInference = async () => {
    if (!draftBBox) {
        alert("Please draw a target region on the map first!");
        return;
    }
    
    setPatchLoading(true);
    // Construct true standard geographic array boundaries
    const lonMin = Math.min(draftBBox.start[0], draftBBox.end[0]);
    const lonMax = Math.max(draftBBox.start[0], draftBBox.end[0]);
    const latMin = Math.min(draftBBox.start[1], draftBBox.end[1]);
    const latMax = Math.max(draftBBox.start[1], draftBBox.end[1]);
    
    const bbox = [lonMin, latMin, lonMax, latMax];
    
    const centerLon = (lonMin + lonMax) / 2;
    const centerLat = (latMin + latMax) / 2;
    
    const result = await fetchPatchInference(bbox, 10, dateRange); 
    setPatchLoading(false);
    
    if (result && result.points) {
       setData(result.points);
       setClusters(result.clusters || []);
       
       if (result.points.length > 0) {
           setViewState(v => ({
             ...v,
             longitude: centerLon,
             latitude: centerLat,
             zoom: 12,
             transitionDuration: 2500,
             transitionInterpolator: new FlyToInterpolator()
           }));
       } else {
           alert(`Live Inference Complete: No Debris detected in the last ${dateRange.replace("last_", "").replace("_", " ")} around this custom tracking patch!`);
       }
    } else {
       alert("Live Patch Inference failed to connect to Sentinel Hub Simulation.");
    }
  };

  const loadData = async () => {
    const res = await getVisualizationData();
    if (res && res.points) {
      setData(res.points);
      
      // Auto center map on first point if available
      if (res.points.length > 0) {
        setViewState(v => ({
          ...v,
          longitude: res.points[0].lon,
          latitude: res.points[0].lat
        }));
      }
    }
  };

  useEffect(() => {
    loadData();
    
    // Auto-refresh every 5 seconds to simulate real-time
    const interval = setInterval(() => {
      loadData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const result = await predictDebris(file);
    setLoading(false);

    if (result && result.points) {
      setData(result.points);
      if (result.points.length > 0) {
         setViewState(v => ({
           ...v,
           longitude: result.points[0].lon,
           latitude: result.points[0].lat,
           zoom: 14,
           transitionDuration: 3000,
           transitionInterpolator: new FlyToInterpolator()
         }));
      } else {
         alert("Inference Complete: No Marine Plastics / Debris were detected in this image!");
      }
    } else {
       alert("Error processing the image or connecting to the local U-Net framework.");
    }
  };

  const layers = [
    new PolygonLayer({
      id: 'background-catcher',
      data: [{polygon: [[-180, 90], [180, 90], [180, -90], [-180, -90]]}],
      getPolygon: d => d.polygon,
      getFillColor: [0,0,0,0],
      pickable: true,
      visible: isDrawing
    }),
    new PolygonLayer({
      id: 'selection-boundary',
      data: draftBBox ? [{
        polygon: [
          [draftBBox.start[0], draftBBox.start[1]],
          [draftBBox.end[0], draftBBox.start[1]],
          [draftBBox.end[0], draftBBox.end[1]],
          [draftBBox.start[0], draftBBox.end[1]]
        ]
      }] : [],
      getPolygon: d => d.polygon,
      getFillColor: [0, 120, 255, 60],
      getLineColor: [0, 80, 255, 255],
      lineWidthMinPixels: 2,
      stroked: true,
      filled: true
    }),
    new HeatmapLayer({
      id: 'heatmap-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getWeight: d => d.probability,
      radiusPixels: 25,
      intensity: 3,
      threshold: 0.05
    }),
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      getPosition: d => [d.lon, d.lat],
      getFillColor: d => [255, 200, 0, 200], // Yellow for points
      getRadius: d => 50,
      radiusMinPixels: 2,
      radiusMaxPixels: 10,
    }),
    new ScatterplotLayer({
      id: 'cluster-layer',
      data: clusters,
      getPosition: d => d.center,
      getFillColor: [0, 255, 200, 150],
      getLineColor: [0, 255, 200, 255],
      lineWidthMinPixels: 2,
      stroked: true,
      getRadius: d => Math.max(90, d.density * 6), // Scale circle dynamically to debris density
    })
  ];

  return (
    <div className="vis-container">
      <div className="vis-header">
        <div>
          <h2>Marine Plastic Hotspots</h2>
          <p>Real-time detection overlay (Heatmap + Points)</p>
        </div>
        
        <div className="vis-controls" style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
          
          <div className="patch-controls" style={{ display: 'flex', gap: '8px', background: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '8px', border: '1px solid #333' }}>
            <div style={{color: '#aaa', fontSize: '13px', alignSelf: 'center', marginRight: '5px'}}>
               <Target size={14} style={{verticalAlign: 'text-bottom', marginRight: '4px'}}/>
               Live Region Focus
            </div>
            <button className="btn-primary" onClick={() => { setIsDrawing(!isDrawing); }} style={{background: isDrawing ? '#ff4444' : '#666', border: '1px solid #444'}}>
               {isDrawing ? 'Cancel Drawing' : 'Draw Patch Area'}
            </button>
            <select value={dateRange} onChange={e => setDateRange(e.target.value)} style={{background: '#111', color: 'white', border: '1px solid #444', borderRadius: '4px', padding: '4px 8px'}}>
               <option value="last_1_day">Last 24 Hours</option>
               <option value="last_3_days">Last 3 Days</option>
               <option value="last_5_days">Last 5 Days</option>
            </select>
            <button className="btn-primary" onClick={handleRunPatchInference} disabled={patchLoading || loading || !draftBBox} style={{background: !draftBBox ? '#444' : '#0066cc'}}>
               {patchLoading ? 'Scanning...' : 'Run Regional Detection'}
            </button>
          </div>

          <div className="upload-button-wrapper">
            <button className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#3D9970' }}>
              <Upload size={18} />
              {loading ? 'Processing...' : 'Upload .TIF Directly'}
            </button>
            <input 
              type="file" 
              accept=".tif,.tiff,.jpg,.jpeg,.png"
              onChange={handleFileUpload} 
              disabled={loading || patchLoading}
            />
          </div>
        </div>
      </div>

      <div className="map-container">
        <DeckGL
          initialViewState={viewState}
          controller={{dragPan: !isDrawing}}
          layers={layers}
          onViewStateChange={({viewState}) => setViewState(viewState)}
          onDragStart={(info) => {
             if (isDrawing && info.coordinate) {
                setDraftBBox({ start: info.coordinate, end: info.coordinate });
             }
          }}
          onDrag={(info) => {
             if (isDrawing && draftBBox && info.coordinate) {
                setDraftBBox(prev => ({ ...prev, end: info.coordinate }));
             }
          }}
          onDragEnd={(info) => {
             if (isDrawing && draftBBox && info.coordinate) {
                setDraftBBox(prev => ({ ...prev, end: info.coordinate }));
                setIsDrawing(false); 
             }
          }}
          getCursor={({isHovering, isDragging}) => {
             if (isDrawing) return 'crosshair';
             if (isDragging) return 'grabbing';
             if (isHovering) return 'pointer';
             return 'grab';
          }}
        >
          <Map mapStyle={MAP_STYLE} />
        </DeckGL>
      </div>

      {isDrawing && (
        <div style={{
          position: 'absolute', 
          bottom: '20px', 
          left: '50%', 
          transform: 'translateX(-50%)',
          background: 'rgba(0,100,255,0.9)', 
          color: 'white', 
          padding: '10px 20px', 
          borderRadius: '30px',
          zIndex: 1000,
          boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
          pointerEvents: 'none',
          fontWeight: 'bold'
        }}>
          DRAG MOUSE TO DRAW TARGET PATCH
        </div>
      )}
    </div>
  );
};

export default Visualization;
