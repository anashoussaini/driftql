import jax
import jax.numpy as jnp
import time

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# 1. Use larger arrays (10,000 x 10,000 is ~400MB per matrix)
size = 10000
print(f"\n1. Creating {size}x{size} matrices...")
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (size, size))
y = jax.random.normal(key2, (size, size))

# Force JAX to instantiate the arrays on the GPU before moving on
x.block_until_ready()
y.block_until_ready()
print("✓ Arrays loaded to GPU.")

# 2. Define the heavy operation
@jax.jit
def heavy_matmul(a, b):
    return jnp.dot(a, b)

# 3. Compile the function
print("\n2. Compiling (JIT)...")
_ = heavy_matmul(x, y).block_until_ready() # The first run triggers compilation
print("✓ Compiled.")

# 4. Stress the GPU compute cores
print("\n3. Stressing GPU (check nvidia-smi now!)...")
start_time = time.time()

iterations = 50
for i in range(iterations):
    result = heavy_matmul(x, y)

# CRITICAL: This forces Python to wait until all 50 loops finish on the GPU
result.block_until_ready()

end_time = time.time()
print(f"✓ Completed {iterations} heavy operations in {end_time - start_time:.2f} seconds.")