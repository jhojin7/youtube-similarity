from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from pathlib import Path
import json

def generate_transcript(video_id:str):
    result = YouTubeTranscriptApi.get_transcript(video_id, languages='en ko en'.split())
    result_concat = '\n\n'.join([res["text"] for res in result])
    return result_concat



if __name__=="__main__":
    sentencetansformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    crossencoder_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    video_ids_dict = json.load(Path(".data","video_ids.json").open('r'))
    video_ids = list(video_ids_dict.values())[1]

    for _id in video_ids:
        transcript_p = Path(Path(),"transcript",f"{_id}.txt")
        if transcript_p.exists():
            print(_id, "exists")
            continue
        print(_id)
        res = generate_transcript(_id)
        transcript_p.parent.mkdir(exist_ok=True, parents=True)
        transcript_p.write_text(res,encoding='utf-8')
    

    for i in range(len(video_ids)):
        for j in range(i+1,len(video_ids)):
            script1 = Path(Path(),"transcript",f"{video_ids[i]}.txt").read_text(encoding='utf-8')
            script2 = Path(Path(),"transcript",f"{video_ids[j]}.txt").read_text(encoding='utf-8')

            ######### cosine similarity
            embeddings1 = sentencetansformer_model.encode(script1)
            embeddings2 = sentencetansformer_model.encode(script2)
            similarities = sentencetansformer_model.similarity(embeddings1, embeddings2)
            # ######### cross-encoder
            pred = crossencoder_model.predict([[script1, script2]])

            print(">>>", i,video_ids[i], j,video_ids[j], "cosine",similarities, "cross-encoder",pred)
