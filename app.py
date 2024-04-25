from flask import Flask, request, jsonify
import torch

from model import RecommendationModel, RetailDataset, load_model

app = Flask(__name__)

# Load dataset for user and item indices
dataset = RetailDataset('cleaned_data.csv')

# Choose the device based on availability of GPU or MPS (Apple's Metal Performance Shaders)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load model for inference
model = RecommendationModel(len(dataset.users), len(dataset.items))
load_model(model, 'recommendation_model.pth')
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move model to the correct device

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if user_id not in dataset.user_to_index:
            return jsonify({'error': 'User not found'}), 404
        
        user_idx = dataset.user_to_index[user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)

        # Compute scores for all items
        scores = {}
        for item, idx in dataset.item_to_index.items():
            item_tensor = torch.tensor([idx], dtype=torch.long, device=device)
            score = model(user_tensor, item_tensor)
            scores[item] = score.item()

        # Select the top 5 recommended items
        recommended_items = sorted(scores, key=scores.get, reverse=True)[:5]

        return jsonify({'recommended_items': recommended_items})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set to False in production