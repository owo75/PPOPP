# PPOPP
Usage: ./PPOPP <m> <n> <k> <rank> <eps> <spectrum_type> <s_max> <s_min>
  <spectrum_type>: 'geometric', 'geometric_zero', 'uniform', 'cluster0', 'cluster1', 'arithmetic', 'normal'
  <s_max>: Maximum singular value (e.g., 1.0)
  <s_min>: Minimum singular value (e.g., 1e-10)
./PPOPP 16384 16384 1024 32 1e-6 geometric 1 1e-6 --debug


