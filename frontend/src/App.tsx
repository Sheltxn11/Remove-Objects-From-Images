import React, { useState, ChangeEvent } from 'react';
import { Upload, SplitSquareHorizontal, Eraser, ImageIcon, Wand2 } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000';

function App() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setOriginalImage(event.target?.result as string);
        setProcessedImage(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!originalImage) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/process_image`, {
        image: originalImage
      });
      
      if (response.data.error) {
        setError(response.data.error);
      } else if (response.data.processed_image) {
        setProcessedImage(response.data.processed_image);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while processing the image');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setOriginalImage(event.target?.result as string);
        setProcessedImage(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen gradient-bg">
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center mb-8 float-animation">
          <div className="inline-flex items-center gap-3 mb-4">
            <Wand2 className="w-8 h-8 text-white" />
            <h1 className="text-4xl font-bold text-gradient">Interior Magic</h1>
          </div>
          <p className="text-gray-300 text-lg">Transform your space by removing plants with AI</p>
        </div>

        <div className="glass-effect hover-lift rounded-2xl p-8 mb-8">
          <div className="flex flex-col items-center gap-8">
            <label 
              className={`w-full max-w-md flex flex-col items-center px-6 py-8 rounded-xl border-2 border-dashed 
                cursor-pointer transition-all duration-300 ${
                isDragging 
                  ? 'border-white bg-white/10' 
                  : 'border-gray-400/50 hover:border-white hover:bg-white/5'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className={`w-12 h-12 mb-3 transition-all duration-300 ${
                isDragging ? 'text-white scale-110' : 'text-gray-400'
              }`} />
              <span className={`text-lg mb-2 transition-colors duration-300 ${
                isDragging ? 'text-white' : 'text-gray-300'
              }`}>
                Drop your interior image here
              </span>
              <span className="text-sm text-gray-400">or click to browse</span>
              <input type="file" className="hidden" onChange={handleImageUpload} accept="image/*" />
            </label>

            <div className="w-full max-w-md">
              <div className="flex flex-col gap-3">
                {error && (
                  <div className="text-red-500 bg-red-100/10 p-3 rounded-lg text-sm">
                    {error}
                  </div>
                )}
                <button
                  onClick={processImage}
                  disabled={!originalImage || isProcessing}
                  className="w-full px-6 py-3 bg-white text-black rounded-xl
                    hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2
                    transition-all duration-300 shadow-lg hover:shadow-white/10 disabled:shadow-none"
                >
                  <Eraser className="w-5 h-5" />
                  {isProcessing ? 'Removing Plants...' : 'Remove Plants'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {originalImage && (
          <div className="glass-effect hover-lift rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <SplitSquareHorizontal className="w-6 h-6 text-white" />
              <h2 className="text-2xl font-semibold text-gradient">
                Comparison View
              </h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="flex flex-col items-center">
                <div className="mb-3 flex items-center gap-2">
                  <ImageIcon className="w-5 h-5 text-gray-300" />
                  <span className="font-medium text-gray-300">Original Image</span>
                </div>
                <div className="w-full overflow-hidden rounded-xl shadow-lg transition-transform duration-300 hover:scale-[1.02]">
                  <img
                    src={originalImage}
                    alt="Original"
                    className="w-full h-auto"
                  />
                </div>
              </div>
              
              <div className="flex flex-col items-center">
                <div className="mb-3 flex items-center gap-2">
                  <Eraser className="w-5 h-5 text-gray-300" />
                  <span className="font-medium text-gray-300">Processed Image</span>
                </div>
                {processedImage ? (
                  <div className="w-full overflow-hidden rounded-xl shadow-lg transition-transform duration-300 hover:scale-[1.02]">
                    <img
                      src={processedImage}
                      alt="Processed"
                      className="w-full h-auto"
                    />
                  </div>
                ) : (
                  <div className="w-full min-h-[300px] bg-black/20 rounded-xl border border-gray-700 
                    flex items-center justify-center text-gray-400">
                    {isProcessing ? (
                      <div className="flex flex-col items-center gap-3">
                        <div className="relative w-12 h-12">
                          <div className="absolute top-0 left-0 w-full h-full border-4 border-gray-700 rounded-full"></div>
                          <div className="absolute top-0 left-0 w-full h-full border-4 border-white rounded-full 
                            border-t-transparent animate-spin"></div>
                        </div>
                        <span className="text-white font-medium">Processing image...</span>
                      </div>
                    ) : (
                      <span>Processed image will appear here</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;