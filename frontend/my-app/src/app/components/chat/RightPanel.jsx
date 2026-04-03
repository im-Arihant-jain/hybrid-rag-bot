"use client";


import React from 'react';
import ProductCard from './ProductCard';
import images from "./assets/images.png";
 

const RightPanel = ({ evalstatefunc, messages }) => {
    
  return (
    <div style={{
      overflow: 'auto',
      scrollbarWidth: 'none', /* Firefox */
      msOverflowStyle: 'none', /* IE and Edge */
    }} className="flex flex-col items-center p-4 w-full h-full rounded-lg shadow-md ">
      {/* <div className="flex flex-col w-full md:w-1/4"> */}
      <ProductCard
        title="Designed with ❤ by Arihant Jain"
        description="Upload the pdf and ask anything about it ..."
        image={images}  // Replace with the actual image path
        price="₹2000"
      />
      <br/>
      <br/>
      <button
      onClick={() => evalstatefunc(true)}
      className="border p-2 border-white rounded-md hover:bg-gray-300 text-lg font-bold text-white-800 mb-2">EVALUATE</button>

      {/* </div> */}
    </div>
  );
};

export default RightPanel;
