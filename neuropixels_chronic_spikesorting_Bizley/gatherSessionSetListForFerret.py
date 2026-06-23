def gatherSessionSetListForFerret(ferret):
    ### the goal of this function will be to take as input a ferret and to output at minimum the
    ### list of all three-week-long session sets. Those session set names will themselves be used as input
    ### into a function which pulls specific parameters. The idea is you first run this, then you get a list
    ### which you can iterate through, then you iterate through the list grabbing the specific parameters
    ### as you need them.
    ### I think that I will not actually be making the names here particularly informative though
    ### because it is difficult to do so here.. Informative names may be in the parameters if at all.
    ### I'll try to put brief comments next to each though. This is also the only reason I am doing
    ### this manually instead of with a loop.
    #### Comment out ones you want to ignore.
    if ferret == 'F2302_Challah':
        recordingZones = {'SessionSet01', ### first set, May 2024
                          'SessionSet02', ### second set, June 2024
                          'SessionSet03', ### third set, July 2024. Some PFC problems, not an overwhelming amount.
                          'SessionSet04', #/# ### July into August. Pulse problems begin. If I want the good PFC data I need this dealt with. Or I need to figure out how to process the PFC data well. I have marked sets where I think the PFC data is tolerable but I have this issue with "##". There are four. So I have a lot of data to gain from it.
                          'SessionSet05', #/# ### August into sept
                          'SessionSet06', #/# ### sept into oct
                          'SessionSet07', #/# NOTONNAS ### oct into nov. Also apparently this set isn't on the nas, should deal with that
                          # 'SessionSet08', ### nov into dec, not on harddrive This is also approximately where I give up hope on good PFC data
                          # 'SessionSet09', ### Jan into feb, includes 4 weeks and a gap?
                          # 'SessionSet10', ### pulse issue gone! Intermittant freezing starts though. Feb to March
                          # 'SessionSet11', ### Should be a good session except PFC is bad? March through April
                          # 'SessionSet12', ### Middle spout is saturating apparently? Oh yeah, this is when that wire was open...
                          # 'SessionSet13', ### June 2025, seems good except I don't like PFC data
                          # 'SessionSet14',
                          # 'SessionSet15',
                          # 'SessionSet16', ### data is generally cleaner in this zone but the PFC is busted. Whether this is a trove or not really depends on if we can deal with that noise. And we might be able to! But.
                          # 'SessionSet17',
                          # 'SessionSet18',
                          # 'SessionSet19',
                          # 'SessionSet20', ### this is the final one, I think it is my pilot data with correlated/uncorrelated noise.

                          }
    elif ferret == 'F2301_Boule':
        recordingZones = {
                          'SessionSet01',#/# ### First set Dec 2024. Pulse problems present
                          'SessionSet02',#/#
                          'SessionSet03',#/# ### Pulse problems solved in middle of this.
                          # 'SessionSet04', ### intermittent freezes. Not sure how much this is or is not a problem.
                          'SessionSet05', ### fully good apparently. Might potentially also include the april 28th 2025 week but that week has a saturating middle spout problem and honestly, there really is no dealing with that
                          'SessionSet06', ### another fully good week, June 2025
                          'SessionSet07', ### includes very short term headstage problem.
                          'SessionSet08', ### reportedly, fully good.
                          'SessionSet09', ### last one before AC probe breaks. Only two weeks of recordings before that happens.  Her PFC continues to be wonderful; this is approximately where we start training her on a new talker.
                          }
    elif ferret == 'F1903_Trifle':
        pass
    elif ferret == 'F2105_Clove':
        pass
    return recordingZones