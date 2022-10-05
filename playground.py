# %%
from model.inception3d import *
import torch.nn as nn

# # %%
# sm = torch.jit.script(
#     torch.nn.MaxPool3d(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
# )
# print("MaxPool3D compiled", sm)
# # %%
# sm = torch.jit.script(
#     MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
# )
# print("MaxPool3DSamePadding compiled", sm)
# # %%
# sm.save("maxpool3dsamepadding.pt")
# print("Model saved successfully")

# # %%
# mdl = torch.jit.load("maxpool3dsamepadding.pt")
# print("Model reloaded successfully", mdl)

# # %%
# sm = torch.jit.script(
#     Unit3D(
#         in_channels=3,
#         output_channels=64,
#         kernel_shape=[7, 7, 7],
#         stride=(2, 2, 2),
#         padding=(3, 3, 3),
#         name="whatever",
#     )
# )
# print("Unit3D compiled", sm)
# # %%
# sm.save("unit3d.pt")
# print("Model saved successfully")

# # %%
# mdl = torch.jit.load("unit3d.pt")
# print("Model reloaded successfully", mdl)

# # %%
# sm = torch.jit.script(
#     InceptionModule(
#         in_channels=192, out_channels=[64, 96, 128, 16, 32, 32], name="doesnotmatter"
#     )
# )
# print("Inception Module compiled", sm)
# # %%
# sm.save("inception_module.pt")
# print("Model saved successfully")

# # %%
# mdl = torch.jit.load("inception_module.pt")
# print("Model reloaded successfully", mdl)

# %%
mdl = InceptionI3d(400, in_channels=3)

module_dict = nn.ModuleDict(mdl._modules)

print(module_dict)

sm = torch.jit.script(nn.ModuleDict(mdl._modules))
print(sm)


# %%
sm = torch.jit.script(InceptionI3d(400, in_channels=3))
print("Inception I3D compiled", sm)
# %%
sm.save("inception_i3d.pt")
print("Model saved successfully")

# %%
mdl = torch.jit.load("inception_i3d.pt")
print("Model reloaded successfully", mdl)

# %%
# We'll use the weights from the general pre-trained inception model, then the weights for the fine tuned one on ASL
# Download these from https://github.com/dxli94/WLASL#training-and-testing
LABEL_MAPPING_PATH = "./data_processing/wlasl_class_list.txt"
ID3_PRETRAINED_WEIGHTS_PATH = "./models/WLASL/weights/rgb_imagenet.pt"
WLASL_PRETRAINED_WEIGHTS_PATH = "./models/WLASL/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt"
NUM_CLASSES = 100


def load_inception_model(device=0):
    """
    Args:
        device: int
    Returns:
        pretrained_i3d_model: InceptionI3d
    """

    # Initialize model
    pretrained_i3d_model = torch.jit.script(InceptionI3d(100, in_channels=3))

    # # Load the general inception model weights
    # pretrained_i3d_model.load_state_dict(
    #     torch.load(ID3_PRETRAINED_WEIGHTS_PATH), strict=False
    # )

    # Adapt the final layer for the number of classes we expect
    # pretrained_i3d_model.replace_logits(NUM_CLASSES)

    # Load the weights for the fine-tuned model on ASL
    pretrained_i3d_model.load_state_dict(
        torch.load(WLASL_PRETRAINED_WEIGHTS_PATH, map_location=torch.device("cpu")),
        strict=False,
    )

    # Move to GPU
    # i3d.cuda(device=device)

    # Add data parallelism layer (but this is not actually that useful here since we're only using 1 example)
    # pretrained_i3d_model = nn.DataParallel(pretrained_i3d_model)

    # Put model in inference mode
    pretrained_i3d_model.eval()

    # pretrained_model = PL_resnet50.load_from_checkpoint(checkpoint_path=weight_loc).eval()#.cuda(device=0)
    return pretrained_i3d_model


# %%
pretrained_model = load_inception_model()
# %%
pretrained_model.save("inception_i3d.pt")

# %%
