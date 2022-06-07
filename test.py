import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(
    url = '192.168.1.101:8001',
    verbose = True,
)

print(triton_client)