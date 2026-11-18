# Member 4 implementation checks

- Independent Adam versus `torch.optim.Adam`: maximum error `0`.
- No-slow-EMA ablation versus independent Adam: maximum error `0`.
- AdEMAMix state save/restore continuation: maximum error `0`.
- Author reference inspected: `apple/ml-ademamix@eb517ed3bbc09227e5d062b682723543f6c71f1a`.
- The slow EMA is not bias corrected, matching the paper and author implementation.
