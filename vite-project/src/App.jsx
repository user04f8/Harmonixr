import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import InteractiveGraph from './templates/InteractiveGraph';

function App() {
  const [graph, setGraph] = useState('/plots/tsne_pieces.htm'); // Default graph

  return (
    <div className="min-h-screen flex justify-center items-center bg-gray-100">
      <div className="w-full h-full p-6 bg-white shadow-md rounded-lg">
        <h1 className="text-2xl font-semibold text-center mb-4">Interactive Graphs</h1>

        <div className="flex justify-center mb-4">
          <button
            className="px-4 py-2 mx-2 bg-blue-500 text-white rounded"
            onClick={() => setGraph('/plots/tsne_pieces.htm')}
          >
            Pieces
          </button>
          <button
            className="px-4 py-2 mx-2 bg-blue-500 text-white rounded"
            onClick={() => setGraph('/plots/tsne_clusters.htm')}
          >
            Clusters
          </button>
        </div>

        <InteractiveGraph src={graph} />
      </div>
    </div>
  );
}

export default App
