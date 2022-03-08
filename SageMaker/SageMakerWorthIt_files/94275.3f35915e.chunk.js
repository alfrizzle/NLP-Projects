(self.webpackChunklite=self.webpackChunklite||[]).push([[94275],{14603:(e,n,i)=>{"use strict";i.d(n,{x:()=>r});var a=i(319),l=i.n(a),t=i(61243),d=i(37205),o={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherFollowersCount_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherFollowerCount_publisher"}}]}}].concat(l()(d.j.definitions))},m=i(1279),s=i(84492),r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"AuthorInfo_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"imageId"}},{kind:"Field",name:{kind:"Name",value:"socialStats"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"followerCount"}}]}},{kind:"Field",name:{kind:"Name",value:"customStyleSheet"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherSidebarFollows_customStyleSheet"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherName_publisher"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherFollowersCount_publisher"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherDescription_publisher"}},{kind:"FragmentSpread",name:{kind:"Name",value:"FollowAndSubscribeButtons_user"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherSidebarFollows_user"}}]}}].concat(l()(t.qy.definitions),l()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherName_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}}]}}]),l()(o.definitions),l()(m.m.definitions),l()(s.i.definitions),l()(t.FB.definitions))}},1279:(e,n,i)=>{"use strict";i.d(n,{m:()=>a});var a={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherDescription_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"description"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"bio"}}]}}]}}]}},14337:(e,n,i)=>{"use strict";i.d(n,{v:()=>o});var a=i(319),l=i.n(a),t=i(84683),d=i(27048),o={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherAvatar_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionAvatar_collection"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserAvatar_user"}}]}}]}}].concat(l()(t.d.definitions),l()(d.W.definitions))}},30826:(e,n,i)=>{"use strict";i.d(n,{G:()=>d});var a=i(67294),l=i(71652),t=i(17193),d=function(e){var n=e.link,i=void 0!==n&&n,d=e.scale,o=void 0===d?"M":d,m=e.publisher,s=e.withHalo,r=void 0===s||s;switch(m.__typename){case"User":return a.createElement(t.Yt,{link:i,scale:o,user:m,withHalo:r});case"Collection":return a.createElement(l.v,{link:i,size:t.wC[o],collection:m});default:return null}}},89199:(e,n,i)=>{"use strict";i.d(n,{b:()=>a});var a={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherFollowingCount_publisher"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"socialStats"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"followingCount"}},{kind:"Field",name:{kind:"Name",value:"collectionFollowingCount"}}]}},{kind:"Field",name:{kind:"Name",value:"followedCollections"}},{kind:"Field",name:{kind:"Name",value:"username"}}]}}]}}]}},12549:(e,n,i)=>{"use strict";i.d(n,{gp:()=>c,DX:()=>v,b5:()=>S});var a=i(28655),l=i.n(a),t=i(71439),d=i(67294),o=i(25735),m=i(93310),s=i(87691),r=i(21372),u=i(50458);function k(){var e=l()(["\n  fragment PublisherFollowingCount_publisher on Publisher {\n    __typename\n    id\n    ... on User {\n      socialStats {\n        followingCount\n        collectionFollowingCount\n      }\n      followedCollections\n      username\n    }\n  }\n"]);return k=function(){return e},e}var c=function(e){var n,i,a,l,t=(0,o.VB)({name:"enable_fix_follow_counts",placeholder:!1}),d=null!==(n="Collection"===e.__typename?0:(null===(i=e.socialStats)||void 0===i?void 0:i.followingCount)+e.followedCollections)&&void 0!==n?n:0;return t&&"User"===e.__typename&&(d=(null===(a=e.socialStats)||void 0===a?void 0:a.followingCount)+(null===(l=e.socialStats)||void 0===l?void 0:l.collectionFollowingCount)),{followingCount:d,isFollowingCountVisible:d>0}},v=function(e){var n,i=e.publisher,a=e.linkStyle,l=void 0===a?"SUBTLE":a,t=c(i),o=t.followingCount,k=t.isFollowingCountVisible,v="User"===i.__typename?(0,u.MzF)(null!==(n=i.username)&&void 0!==n?n:""):"",S=!!v;if(!k)return null;var p="".concat((0,r.pY)(o)," Following");return S?d.createElement(m.r,{linkStyle:l,href:v},p):d.createElement(s.F,{tag:"span",scale:"L",color:"DARKER"},p)},S=(0,t.Ps)(k())},61243:(e,n,i)=>{"use strict";i.d(n,{qy:()=>r,FB:()=>u});var a=i(319),l=i.n(a),t=i(89199),d=i(68216),o=i(14337),m=i(54341),s=i(77136),r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherSidebarFollows_customStyleSheet"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"CustomStyleSheet"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"blogroll"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"visibility"}}]}}]}}]},u={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherSidebarFollows_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"username"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherFollowingCount_publisher"}},{kind:"FragmentSpread",name:{kind:"Name",value:"userUrl_user"}}]}}].concat(l()(t.b.definitions),l()(d.$m.definitions))},k={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PublisherSidebarFollows_followedEntity"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Publisher"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherAvatar_publisher"}}]}}].concat(l()(o.v.definitions))};[{kind:"OperationDefinition",operation:"query",name:{kind:"Name",value:"PublisherSidebarFollowsQuery"},variableDefinitions:[{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"userId"}},type:{kind:"NonNullType",type:{kind:"NamedType",name:{kind:"Name",value:"ID"}}}},{kind:"VariableDefinition",variable:{kind:"Variable",name:{kind:"Name",value:"limit"}},type:{kind:"NamedType",name:{kind:"Name",value:"Int"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"userFollows"},arguments:[{kind:"Argument",name:{kind:"Name",value:"userId"},value:{kind:"Variable",name:{kind:"Name",value:"userId"}}},{kind:"Argument",name:{kind:"Name",value:"limit"},value:{kind:"Variable",name:{kind:"Name",value:"limit"}}}],selectionSet:{kind:"SelectionSet",selections:[{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"hasDomain"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserMentionTooltip_user"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherSidebarFollows_followedEntity"}}]}},{kind:"InlineFragment",typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionTooltip_collection"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PublisherSidebarFollows_followedEntity"}}]}}]}}]}}].concat(l()(m.O.definitions),l()(k.definitions),l()(s.g.definitions))},77136:(e,n,i)=>{"use strict";i.d(n,{g:()=>o});var a=i(319),l=i.n(a),t=i(84683),d=i(19308),o={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"CollectionTooltip_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"description"}},{kind:"Field",name:{kind:"Name",value:"subscriberCount"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionAvatar_collection"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionFollowButton_collection"}}]}}].concat(l()(t.d.definitions),l()(d.Iq.definitions))}},75210:(e,n,i)=>{"use strict";i.d(n,{L:()=>s});var a=i(67294),l=i(71652),t=i(82405),d=i(77355),o=i(20113),m=i(87691),s=function(e){var n=e.collection,i=e.buttonSize,s=e.buttonStyleFn,r=n.name,u=n.description;return a.createElement(d.x,{padding:"15px",display:"flex",flexDirection:"column",width:"300px"},a.createElement(d.x,{display:"flex",flexDirection:"row",justifyContent:"space-between",whiteSpace:"normal",borderBottom:"BASE_LIGHTER",paddingBottom:"10px",marginBottom:"10px"},a.createElement(d.x,{display:"flex",flexDirection:"column",paddingRight:"5px"},a.createElement(o.X6,{scale:"S"},r),a.createElement(m.F,{scale:"S"},u)),a.createElement(d.x,null,a.createElement(l.v,{collection:n,link:!0}))),a.createElement(d.x,{display:"flex",flexDirection:"row",alignItems:"center",justifyContent:"space-between"},a.createElement(m.F,{scale:"M"},"Followed by ",n.subscriberCount," people"),a.createElement(t.Fp,{collection:n,simpleButton:!0,buttonSize:i,buttonStyleFn:s,susiEntry:"follow_card"})))}},84492:(e,n,i)=>{"use strict";i.d(n,{i:()=>o});var a=i(319),l=i.n(a),t=i(78693),d=i(71069),o={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"FollowAndSubscribeButtons_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"UserFollowButton_user"}},{kind:"FragmentSpread",name:{kind:"Name",value:"UserSubscribeButton_user"}}]}}].concat(l()(t.s.definitions),l()(d.w.definitions))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/94275.3f35915e.chunk.js.map