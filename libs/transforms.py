import torch
import torch_dct as dct
import libs.dct_butterfly_torch as dct_b
import argparse

def apply_luma_transform(args: argparse.Namespace, blocks: torch.Tensor, dwt_model=None) -> torch.Tensor:
    """
    Applies the selected discrete transform to the luma blocks.
    """
    if args.transform == 'DWT':
        if dwt_model is None:
            raise ValueError("dwt_model cannot be None when transform is DWT")
        yl, yh = dwt_model(blocks.unsqueeze(1).float())
        yh = yh[0]
        top_row = torch.cat((yl, yh[:, :, 0, :, :]), dim=3)
        bottom_row = torch.cat((yh[:, :, 1, :, :], yh[:, :, 2, :, :]), dim=3)
        return torch.cat((top_row, bottom_row), dim=2)
    elif args.transform == 'DCT_B':
        return dct_b.dct_32_2d(blocks.type(torch.int32))
    else:
        return dct.dct_2d(blocks)

def apply_chroma_transform(args: argparse.Namespace, U_blocks: torch.Tensor, V_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the selected discrete transform to the chroma blocks.
    Note: DWT is not currently applied to chroma blocks in EVCA.
    """
    if args.transform == 'DCT_B':
        U_DTs = dct_b.dct_16_2d(U_blocks.type(torch.int32))
        V_DTs = dct_b.dct_16_2d(V_blocks.type(torch.int32))
    else:
        U_DTs = dct.dct_2d(U_blocks)
        V_DTs = dct.dct_2d(V_blocks)
    return U_DTs, V_DTs
