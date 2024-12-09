// src/components/InteractiveGraph.jsx
import React from 'react';
import Iframe from 'react-iframe';

const InteractiveGraph = ({ src }) => {
  return (
    <div className="h-full w-full">
      <Iframe 
        url={src}
        width="100%"
        height="800"
        id="interactive-graph"
        display="initial"
        position="relative"
      />
    </div>
  );
};

export default InteractiveGraph;
