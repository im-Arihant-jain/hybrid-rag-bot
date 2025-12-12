"use client";


import React from 'react';
import ProductCard from './ProductCard';
import images from "./assets/images.png";

import { useRouter } from "next/navigation";

const RightPanel = ({ messages }) => {
   const router = useRouter();

  const handleEvaluate = () => {
    const encoded = encodeURIComponent(JSON.stringify(messages));
    router.push(`/evaluation?data=${encoded}`);
  };
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
       onClick={handleEvaluate}
      className="text-lg font-bold text-white-800 mb-2">EVALUATE</button>

      {/* </div> */}
    </div>
  );
};

export default RightPanel;