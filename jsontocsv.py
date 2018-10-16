try:
    import json
except ImportError:
    import simplejson as json    

with open('DLIndaba2018_hashtag_all_data.json') as data_file:    
    data = json.loads(data_file)  


with 'text' in data: # only messages contains 'text' field is a tweet
    print(tweet['id']) # This is the tweet's id
    print(tweet['created_at']) # when the tweet posted
    print(tweet['text']) # content of the tweet
                        
    print(tweet['user']['id']) # id of the user who posted the tweet
    print(tweet['user']['name']) # name of the user, e.g. "Wei Xu"
    print(tweet['user']['screen_name']) # name of the user account, e.g. "cocoweixu"

    hashtags = []
    for hashtag in tweet['entities']['hashtags']:
        hashtags.append(hashtag['text'])
    print(hashtags)