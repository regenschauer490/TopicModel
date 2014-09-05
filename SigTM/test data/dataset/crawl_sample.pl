use strict;
use warnings;
use YAML::Tiny;
use Encode;
use utf8;
use LWP::UserAgent;
use Net::OAuth;
use Data::Random qw(rand_chars);
use Data::Dumper;
use MyUtility;
use MyTwitter;

#
our $base_pass = "./";
our $save_pass = $base_pass . "raw" ."/";
our $ac_conf = ( YAML::Tiny->read ('conf.yml'))->[0];
#

our $twitter = MyTwitter->new($ac_conf);
my @queries = @{&SimpleReadDataFile($base_pass . "query.txt", "cp932")};

&CrawlTweet(\@queries);

my @users = @{&SimpleReadDataFile($save_pass . "user_ids.txt", "cp932")};
print "user num" . scalar(@users) . "\n";
	
&CrawlTimeline(\@users);
	
END;

sub TweetCrawl
{
	my $qref = shift;
	my @users;

	foreach my $query (@{$qref}){
		my @users_sr;
		my @tweets_sr;
		
		my @local_querys = split(/,/, $query);
		shift(@local_querys);
				 
		foreach my $local_query (@local_querys){
			my $max_id = 900000000000000000;
			my $prev_max_id = $max_id;
			my $loop = 1;
			&Print("q:".$local_query."\n");
			
			while(1){
				RETRY1:
				my $sr = 0;
				my $size = 0;

				eval{
					$sr = $twitter->SearchTweet({q => $local_query." exclude:retweets", lang => "ja", count => 100, max_id => $max_id}); sleep( int(rand 3) + 3 );
					$size = scalar(@{$sr->{statuses}});
					#print Dumper $sr;
				};
				if($@ || $size eq 0){
					print "catch exception : $local_query - $loop\n";
					sleep(60);
					goto RETRY1;
				}
							
				foreach my $e (@{$sr->{statuses}}){
				#	print "edata:".$e->{user}->{id} ."\n";	sleep(1000);
				#	print Dumper $e;
					
					if( &SkipFilter($e->{text}) ){ next; }
					
					my $tyofuku1 = grep{ $_ eq $e->{id} } @tweets_sr;
					if($tyofuku1 != 0){ next; }
					else{ push(@tweets_sr, $e->{id}); }
			
					$max_id = $e->{id};
					
					my $jtmp = &JsonEncode($e);
					&SimpleSaveAddDataFile($jtmp."\n", $save_pass."all_".encode("cp932",$local_querys[0]).".txt", "cp932");
					
					if(!defined($e->{user}->{id})){ next;}
					
					my $tyofuku2 = grep{ $_ eq $e->{user}->{id} } @users_sr;
					if($@){ print "catch!! $@\n"; print Dumper $e; }
					
					if($tyofuku2 != 0){ next; }
					else{ push(@users_sr, $e->{user}->{id}); }
				
					&SimpleSaveAddDataFile($jtmp."\n", $save_pass.encode("cp932",$local_querys[0]).".txt", "cp932" );
				}
				
				&Print($loop."【$size】 ");
				&Print("\n") if($loop % 10 == 0);
				if($prev_max_id eq $max_id && $size != 0){ &Print(" 【$local_query last】\n"); last; }
				if($loop > 100){ &Print(" 【$local_query upper limit】\n"); last; }
				
				$prev_max_id = $max_id;
				++$loop;
			}
		}
		
		my %grep;
		push(@users, @users_sr);
		@users = grep{ ++$grep{$_} < 2 } @users;
		&SimpleSaveArrayDataFile(\@users, $save_pass."user_ids.txt", "cp932" );
		
		print "\n";
	}
}

sub CrawlTimeline
{
	my $uref = shift;
	my $twitter = MyTwitter->new($ac_conf);
	
	foreach my $uid (@{$uref}){
		my $max_id = 900000000000000000;
		my $prev_max_id = $max_id;
		my $loop = 1;
		&Print("user_id:".$uid."\n");
			
		while(1){
			RETRY2:
			my $sr;
			my $size = 0;
			
			eval{			
				$sr = $twitter->StatusUserTimeline({user_id => $uid, lang => "ja", count => 100, max_id => $max_id});	sleep( int(rand 10) + 4 );
				#print Dumper $udata;
				$size = scalar(@$sr);
			};
			if($@ || $size eq 0){
				print "catch exception : $uid\n";
				sleep(60);
				goto RETRY2;
			}
		
			foreach my $e (@{$sr->{statuses}}){	
				if( &SkipFilter($e->{text}) ){ next; }
					
				$max_id = $e->{id};
					
				my $jtmp = &JsonEncode($e);								
				&SimpleSaveAddDataFile($jtmp."\n", $save_pass.encode("cp932",$uid).".txt", "cp932" );
			}
				
			&Print($loop."【$size】 ");
			&Print("\n") if($loop % 10 == 0);
			if($prev_max_id eq $max_id && $size != 0){ &Print(" 【$uid last】\n"); last; }
				
			$prev_max_id = $max_id;
			++$loop;
		}
	}
}

sub SkipFilter
{
	my $text = shift;
	my $skip = 0;
	
	$skip = ($text =~ m/(RT)|(bot)|(自動)|(拡散)|(定期)/);
	
	return $skip;
}