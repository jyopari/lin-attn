from ncu import *

def exp_block_config_AI():
    results_recurrent = []
    results_parallel = []
    
    data_configs = [
        (1, 32, 8, 64), 
        (2, 32, 8, 64),
        (4, 64, 8, 64),
        (4, 128, 8, 64), # for paper
    ]

    for B, L, H, D in data_configs:
        print(f"B: {B}, L: {L}, H: {H}, D: {D}")
        Q = torch.randn(B, L, H, D, device="cuda") / 50.
        K = torch.randn(B, L, H, D, device="cuda") / 50.
        V = torch.randn(B, L, H, D, device="cuda") / 50.

        for BK in [16, 32, 64]:
            for BV in [16, 32, 64]:
                print(f"recurrent BK: {BK}, BV: {BV}")
                stats = fused_recurrent_dataflow(Q, K, V, count=True, BK=BK, BV=BV)
                results_recurrent.append({
                    "batch": B,
                    "length": L,
                    "heads": H,
                    "dim": D,
                    "BK": BK,
                    "BV": BV,
                    **stats,
                })


        for BT in [16, 32]:
            BS = BT
            for BK in [16, 32, 64]:
                for BV in [16, 32, 64]:
                    print(f"parallel BT: {BT}, BS: {BS}, BK: {BK}, BV: {BV}")
                    stats = fused_parallel_dataflow(Q, K, V, count=True, BT=BT, BS=BS, BK=BK, BV=BV)
                    results_parallel.append({
                        "batch": B,
                        "length": L,
                        "heads": H,
                        "dim": D,
                        "BT": BT,
                        "BS": BS,
                        "BK": BK,
                        "BV": BV,
                        **stats,
                    })


    return results_recurrent, results_parallel

if __name__ == "__main__":
    results_recurrent, results_parallel = exp_block_config_AI()
    print(results_recurrent)
    print(results_parallel)

    # Save results to CSV
    import pandas as pd

    # Convert lists to DataFrames
    df_recurrent = pd.DataFrame(results_recurrent)
    df_parallel = pd.DataFrame(results_parallel)

    # # Save to CSV
    # df_recurrent.to_csv("csv/recurrent_results.csv", index=False)
    # df_parallel.to_csv("csv/parallel_results.csv", index=False)


