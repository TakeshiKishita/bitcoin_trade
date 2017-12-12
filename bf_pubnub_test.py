from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
from pubnub.pubnub import PubNub, SubscribeListener

PubNub_Subscribe_Key = 'sub-c-52a9ab50-291b-11e5-baaa-0619f8945a4f'
board = 'lightning_board_snapshot_BTC_JPY'
diff = 'lightning_board_BTC_JPY'
ticker = 'lightning_ticker_BTC_JPY'
executions = 'lightning_executions_BTC_JPY'
channels = [board, diff, ticker, executions]
pnconfig = PNConfiguration()
pnconfig.subscribe_key = PubNub_Subscribe_Key
pnconfig.ssl = False
pubnub = PubNub(pnconfig)

my_listener = SubscribeListener()
pubnub.add_listener(my_listener)
pubnub.subscribe().channels(channels).execute()
my_listener.wait_for_connect()
print('subscribed')

result = my_listener.wait_for_message_on(channels)
print(result.message, flush=True)

pubnub.unsubscribe().channels(channels).execute()
my_listener.wait_for_disconnect()
print('unsubscribed')