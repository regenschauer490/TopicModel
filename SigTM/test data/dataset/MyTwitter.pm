package MyTwitter;

use strict;
use warnings;
use Encode;
use utf8;
use LWP::UserAgent;
use Net::OAuth;
use Data::Random qw(rand_chars);
use Data::Dumper;
use MyUtility;
use AnyEvent::Twitter::Stream;
use Exporter;	# Exporterモジュールを使う

our @ISA    = qw(Exporter);				# Exporterモジュールを継承
our @EXPORT = qw(TwSearchTweet );		# デフォルトでエクスポートするシンボル

our $endpoint_base = "https://api.twitter.com/1.1/";
our $get_format = ".json";

sub new 
{
	my $class = shift;
	my $act = shift;
	
	my $self = {
		ac_token => $act,
		user_agent => LWP::UserAgent->new,
	};
		
	return bless $self, $class;
}

sub _TwitterRequest
{
	my $self = shift;
	my $url = shift;
	my $params = shift;

	my $request = Net::OAuth->request('protected resource')->new(
	    consumer_key => $self->{ac_token}->{'consumer_key'},
	    consumer_secret => $self->{ac_token}->{'consumer_secret'},
		request_url      => $url,
		request_method   => 'GET',
		signature_method => 'HMAC-SHA1',
		timestamp        => time,
		nonce        => join('', rand_chars(ssize => 16, set => 'alphanumeric')),
	    token => $self->{ac_token}->{'access_token'},
	    token_secret => $self->{ac_token}->{'access_token_secret'},
	
		extra_params => $params
	);

	$request->sign;
	
	my $response = $self->{user_agent}->get($request->to_url);
	my $decode = &JsonDecode($response->content);
	
	return $decode;
}

# Public streamsからランダムにtweetを取得
# argment1(option): フィルタ用クエリ(track), 言語指定(lang), ユーザ指定(follow), 位置情報指定(location) [ref hash]
sub RandomSample
{
	my $self = shift;
	my $option = shift;
	
	my $ref = (defined($option) and (exists($option->{track}) or exists($option->{follow}) or exists($option->{location})))
		? $self->_TwitterRequest("https://stream.twitter.com/1.1/"."statuses/filter".$get_format, $option)
		: $self->_TwitterRequest("https://stream.twitter.com/1.1/"."statuses/sample".$get_format, "");
	
	my @result;
	foreach my $e (@{$ref->{statuses}}){
		if(defined($option) and (!exists($option->{lang}) or &is_lang($e, $option->{lang}))){
			push(@result, $e);
		}
		print $e->{text}."\n";
	}
	
	return \@result;
}

#指定tweetをRTしたstatus_idとuser_idを取得
sub GetRetweet
{
	my $self = shift;
	my $status_id = shift;
	
	my $ref = $self->StatusShow({id => $status_id, include_my_retweet => "true", include_entities => "false"});
	my $rts = $ref->{current_user_retweet};
	print Dumper $ref;
	print Dumper $rts;
	
	foreach my $e (@$rts){
		
	}
}

# 指定クエリを含むtweetを検索
# argument1: リクエストパラメータ[ref hash] (例：{q => "HELLSINKER. exclude:retweets", count => 100, max_id => $max_id})
# retuen -> Tweetsオブジェクト [ref]
#
# 詳しくはAPI1.1 https://dev.twitter.com/docs/api/1.1/get/search/tweets
sub SearchTweet
{
	my $self = shift;
	my $params = shift;
	my $url = $endpoint_base."search/tweets".$get_format;
	
	return $self->_TwitterRequest($url, $params);
}

# 指定ユーザ(1人)の情報を取得
# argument1: リクエストパラメータ[ref hash] (例：{screen_name => "Regenschauer490", include_entities => "true"})
# retuen -> Twitterからのレスポンスデータ[ref]
#
# 詳しくはAPI1.1 https://dev.twitter.com/docs/api/1.1/get/users/show
sub UserShow
{
	my $self = shift;
	my $params = shift;
	my $url = $endpoint_base."users/show".$get_format;
	
	return $self->_TwitterRequest($url, $params);
}

# 指定ユーザ(最大100人)の情報を取得
# argument1: リクエストパラメータ[ref hash] (例：{user_id => "123456789,11922960,0120117117", include_entities => "true"})
# retuen -> Twitterからのレスポンスデータ[ref]
#
# 詳しくはAPI1.1 https://dev.twitter.com/docs/api/1.1/get/users/lookup
sub UserLookup
{
	my $self = shift;
	my $params = shift;
	my $url = $endpoint_base."users/lookup".$get_format;
	
	return $self->_TwitterRequest($url, $params);
}

# 指定ユーザのtweetを取得
# argument1: リクエストパラメータ[ref hash] (例：{screen_name => "Regenschauer490", count => 100, max_id => $max_id})
# retuen -> Twitterからのレスポンスデータ[ref]
#
# 詳しくはAPI1.1 https://dev.twitter.com/docs/api/1.1/get/statuses/user_timeline
sub StatusUserTimeline
{
	my $self = shift;
	my $params = shift;
	my $url = $endpoint_base."statuses/user_timeline".$get_format;
	
	return $self->_TwitterRequest($url, $params);
}

# 指定tweetを取得
# argument1: リクエストパラメータ[ref hash] (例：{id => "210462857140252672", include_my_retweet => 1})
# retuen -> Twitterからのレスポンスデータ[ref]
#
# 詳しくはAPI1.1 https://dev.twitter.com/docs/api/1.1/get/statuses/user_timeline
sub StatusShow
{
	my $self = shift;
	my $params = shift;
	my $url = $endpoint_base."statuses/show".$get_format;
	
	return $self->_TwitterRequest($url, $params);
}


## utility

sub is_lang
{
	my $tweet_obj = shift;
	my $lang = shift;
	
	return $tweet_obj->{lang} eq $lang;
}

1;