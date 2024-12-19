Reflection
1. When to use constant memory?
Use constant memory when you have small, read-only data (like a convolution mask) that is accessed by many threads. It’s faster than global memory because of caching and works well when the data stays constant throughout execution.

2. What went well?
I successfully implemented naive and constant memory convolution kernels. Seeing the performance difference between the two helped me understand how CUDA memory types affect speed. I also learned how to handle boundary conditions in convolution.

3. What was challenging?
Debugging memory access issues in CUDA was tricky, especially avoiding out-of-bounds errors. Understanding and optimizing CUDA’s memory hierarchy was also a steep learning curve.

4. What would I do differently?
Next time, I’d plan the memory layout and access patterns better before coding. I’d also explore advanced techniques like shared memory or tiling to boost performance.

5. Anything else?
This assignment taught me a lot about GPU architecture and memory management. It reinforced the importance of careful planning and boundary handling in convolution operations.
