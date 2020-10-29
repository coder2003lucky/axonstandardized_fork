Introduction
====================================

Generating biologically detailed models of neurons is an important goal for modern neuroscience. Unfortunately,
constraining parameters within biologically detailed models can be difficult, leading to poor model predictions,
especially if such models are extended beyond the specific problems for which they were designed. This major obstacle
can be partially overcome by numerical optimization and detailed exploration of parameter space. These processes, which
currently rely on central processing unit (CPU) computation, are computationally demanding, often with exponential
increases in computing time and cost for marginal improvements in model behavior. As a result, models are often
compromised in scale given available CPU-based resources. Here, we present a simulation environment, NeuroGPU, that
takes advantage of the inherent parallelized structure of graphics processing unit (GPU) to accelerate neuronal
simulation. NeuroGPU can simulate most of biologically detailed models from commonly used databases 1-2 orders of
magnitude faster than traditional single core CPU processors, even when implemented on relatively inexpensive GPU
systems. Thus, NeuroGPU offers the ability to apply compartmental, biologically detailed, modeling approaches with
supercomputer-level speed at substantially reduced cost.