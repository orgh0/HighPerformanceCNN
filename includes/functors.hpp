struct add_functor
{
    const float e;
    add_functor(float _e) : e(_e) {}
    __host__ __device__ float operator()(const float &x) const { return x + e; }
};
