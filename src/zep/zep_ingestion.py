import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep
import asyncio
from tqdm import tqdm

async def main():
    # Load environment variables
    load_dotenv()

    # Configure resume offsets: adjust when rerunning to skip already ingested data.
    start_user = 0
    start_thread = 0  # only applies to start_user
    start_msg_idx = 0  # 0-based; only applies to start_user + start_thread

    # Download JSON data
    url = "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json"
    response = requests.get(url)
    data = response.json()
    locomo_df = pd.read_json(url)

    # Save JSON locally
    os.makedirs("./dataset", exist_ok=True)
    with open("./dataset/locomo10.json", "w") as f:
        json.dump(data, f, indent=2)

    print("JSON saved to ./dataset/locomo10.json")
    # Initialize Zep client
    zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")

    # Process each user
    num_users = 10
    max_thread_count = 35

    # Skip users before start_user to avoid duplicate ingestion.
    for graph_idx in range(start_user, num_users):
        conversation = locomo_df['conversation'].iloc[graph_idx]
        graph_id = f"locomo_experiment_user_{graph_idx}"
        print(graph_id)

        try:
            await zep.graph.create(graph_id=graph_id)
        except Exception:
            pass

        thread_start = start_thread if graph_idx == start_user else 0

        for thread_idx in tqdm(range(thread_start, max_thread_count), desc=f"Processing user {graph_idx}"):
            thread_key = f'thread_{thread_idx}'
            legacy_thread_key = f'session_{thread_idx}'

            thread = conversation.get(thread_key)
            key_used = thread_key
            if thread is None:
                thread = conversation.get(legacy_thread_key)
                key_used = legacy_thread_key

            if thread is None:
                continue
            print(key_used)

            msg_start = start_msg_idx if (graph_idx == start_user and thread_idx == start_thread) else 0

            for msg_idx, msg in enumerate(tqdm(thread, desc=f"Processing thread {key_used}")):
                if msg_idx < msg_start:
                    continue
                thread_date_str = conversation.get(f'{key_used}_date_time')
                if thread_date_str is None:
                    continue

                thread_date = thread_date_str + ' UTC'
                date_format = '%I:%M %p on %d %B, %Y UTC'
                date_string = datetime.strptime(thread_date, date_format).replace(tzinfo=timezone.utc)
                iso_date = date_string.isoformat()

                blip_caption = msg.get('blip_captions')
                img_description = f'(description of attached image: {blip_caption})' if blip_caption is not None else ''

                await zep.graph.add(
                    data=msg.get('speaker') +': ' + msg.get('text') + img_description,
                    type='message',
                    created_at=iso_date,
                    graph_id=graph_id,
                )

if __name__ == "__main__":
    asyncio.run(main())
