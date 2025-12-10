
"use client";

import React from 'react';
import MyIcon from './assets/logo';

const ChatArea = ({ messages }) => {
  return (
    <div
      style={{
        overflow: 'auto',
        scrollbarWidth: 'none', /* Firefox */
        msOverflowStyle: 'none',
      }}
      className="flex-1 p-4 bg-gray-200 rounded-lg shadow-md border border-gray-300 overflow-auto"
    >
      <div className="flex w-full items-start m-1">
        
        <div className="shadow-md font-bold flex items-center justify-center text-center p-2 text-gray-800">HYBRID-RAG-BOT</div>
      </div>

      {messages.map((msg, index) => (
        <div key={index} className="flex p-1.5 rounded-lg mb-2">
          <div>
            {/* Display input message */}
            <div className="inline-block p-1.5 rounded-lg mb-2 bg-[#5DE0E6] self-end">
              {msg.input}
            </div>
            <br>
            </br>
            {/* Show loader if output is not present */}
            {msg.output === '' ? (
              <div className="inline-block p-1.5 rounded-lg mb-2 bg-gray-300 self-start animate-pulse">
                Loading...
              </div>
            ) : (
              // Display output message when available
              <div className="inline-block p-1.5 rounded-lg mb-2 bg-[#004AAD] text-white self-start">
                {msg.output}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatArea;