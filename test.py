import torch
from torchvision import transforms
from PIL import Image
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = CNN().to(device)
model.load_state_dict(torch.load("cnn.pth"))
model.eval()

image = Image.open("test_image2.png").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    prediction = "dog." if output.item() > 0.5 else "cat. "

print("It is a ", prediction)